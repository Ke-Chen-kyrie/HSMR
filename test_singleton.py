#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""HSMR 模型单例 + 启动即加载 独立测试脚本.

测两件事, 不依赖 HSMR 完整依赖链 (torch/cv2/detectron2/pyrender 都不用装):

  1) 模型加载是单例模式
     用 AST 从源码里抽出真实的 HSMRModelManager / RenderModelManager 类定义,
     在隔离命名空间里执行 (打桩 load_models 计数), 再 N 线程并发调 get_instance():
       - 并发首调只会真正加载一次 (load_models == 1)
       - 所有线程拿到同一个实例
       - 之后任意次数调用仍是同一实例、不再加载

  2) fastapi 服务起来的时候就要加载
     静态断言 (AST): lifespan 函数里 get_instance()/get_skel_renderer() 出现在 yield 之前
       (uvicorn 必须等 lifespan 完成才收请求, 所以服务一起, 模型已加载常驻)
     黑盒断言 (remote 模式): 未发任何 /infer 前 /health 即 model_loaded:true;
       可选 ssh 数容器日志里 "[启动] 模型已加载" / "[加载]" 出现次数 == 1 次加载.

用法:
  python test_singleton.py unit                       # 单例并发 + 启动加载静态断言 (本地, 毫秒级)
  python test_singleton.py unit --workers 16          # 指定并发线程数 (默认 16)
  python test_singleton.py remote --host 192.168.217.100 --port 8010
  python test_singleton.py remote --host 192.168.217.100 --port 8010 \
          --container hsmr-infer --ssh-cmd "/tmp/hsmr_ssh.sh"   # 追加 docker 日志计数核对
  python test_singleton.py --repo /path/to/repo unit  # 指定工程根 (HSMR-infer/ HSMR-render/ 所在目录)
"""
import argparse
import ast
import base64
import json
import os
import subprocess
import sys
import threading
import urllib.request
from concurrent.futures import ThreadPoolExecutor

DEFAULT_REPO = os.path.dirname(os.path.abspath(__file__))

# 128x128 灰图 PNG (内嵌, 无 PIL 也能做黑盒请求)
TEST_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAIAAAACACAIAAABMXPacAAABLklEQVR4nO3RQREAIAzAMEDm/AuZjDxoFPSu"
    "d2ZOnKcDftcArAFYA7AGYA3AGoA1AGsA1gCsAVgDsAZgDcAagDUAawDWAKwBWAOwBmANwBqANQBrANYArAFY"
    "A7AGYA3AGoA1AGsA1gCsAVgDsAZgDcAagDUAawDWAKwBWAOwBmANwBqANQBrANYArAFYA7AGYA3AGoA1AGsA1g"
    "CsAVgDsAZgDcAagDUAawDWAKwBWAOwBmANwBqANQBrANYArAFYA7AGYA3AGoA1AGsA1gCsAVgDsAZgDcAagDUAa"
    "wDWAKwBWAOwBmANwBqANQBrANYArAFYA7AGYA3AGoA1AGsA1gCsAVgDsAZgDcAagDUAawDWAKwBWAOwBmANwBqA"
    "NQBrANYArAFYA7AGYA3AGoAtFvgCDv+Lo+IAAAAASUVORK5CYII="
)

PASS = "\033[32m[OK]\033[0m"
FAIL = "\033[31m[FAIL]\033[0m"


# ──────────────────────────────────────────────────────────── AST 工具 ────
def load_source(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def extract_class_source(src, class_name):
    """从源码 AST 抽出指定类的源码文本 (ast.unparse), 用于隔离执行."""
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return ast.unparse(node)
    raise ValueError(f"{class_name} 未在源码中找到")


def find_function(tree, name):
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    return None


def _func_dotted(node):
    """ast.Call.func -> 点号链, 如 ['HSMRModelManager','get_instance'];

    也支持以调用结果为底: '_get_skel_renderer().render' → ['_get_skel_renderer','render'].
    """
    names = []
    while isinstance(node, ast.Attribute):
        names.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        names.append(node.id)
        return names[::-1]
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        names.append(node.func.id)
        return names[::-1]
    return None


def call_dotted_in(node, dotted):
    """子树里是否存在指定点号调用, 如 'HSMRModelManager.get_instance'."""
    want = dotted.split(".")
    found = False

    def walk(n):
        nonlocal found
        if found:
            return
        if isinstance(n, ast.Call) and _func_dotted(n.func) == want:
            found = True
            return
        for c in ast.iter_child_nodes(n):
            walk(c)

    walk(node)
    return found


def first_call_pos(fn, dotted):
    """lifespan 内首次出现指定调用的 (lineno,col), 用于与 yield 比先后."""
    want = dotted.split(".")
    pos = None
    for node in ast.walk(fn):
        if isinstance(node, ast.Call) and _func_dotted(node.func) == want:
            p = (node.lineno, node.col_offset)
            if pos is None or p < pos:
                pos = p
    return pos


def first_yield_pos(fn):
    pos = None
    for node in ast.walk(fn):
        if isinstance(node, ast.Yield):
            p = (node.lineno, node.col_offset)
            if pos is None or p < pos:
                pos = p
    return pos


def has_state_runtime_subscript(fn):
    for node in ast.walk(fn):
        if (isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name)
                and node.value.id == "STATE"
                and isinstance(node.slice, ast.Constant) and node.slice.value == "runtime"):
            return True
    return False


def _is_none_compare(test, name):
    return (isinstance(test, ast.Compare) and len(test.ops) == 1
            and isinstance(test.ops[0], ast.Is) and len(test.comparators) == 1
            and isinstance(test.comparators[0], ast.Constant)
            and test.comparators[0].value is None
            and isinstance(test.left, ast.Name) and test.left.id == name)


def assert_double_checked_lock(fn, name):
    """断言: if NAME is None: with _LOCK: if NAME is None: NAME = ..."""
    outer = next(n for n in fn.body
                 if isinstance(n, ast.If) and _is_none_compare(n.test, name))
    withw = next(n for n in ast.walk(outer)
                 if isinstance(n, ast.With)
                 and any(isinstance(it.context_expr, ast.Name)
                         and it.context_expr.id == f"{name}_LOCK"
                         for it in n.items))
    inner = next(n for n in ast.walk(withw)
                 if isinstance(n, ast.If) and _is_none_compare(n.test, name))
    assign = next(n for n in ast.walk(inner)
                  if isinstance(n, ast.Assign)
                  and any(isinstance(t, ast.Name) and t.id == name for t in n.targets))
    return True


# ──────────────────────────────────────────────────────────── 单例并发测试 ────
def run_singleton_concurrent(inference_path, class_name, workers):
    src = load_source(inference_path)
    class_src = extract_class_source(src, class_name)

    ctx = {"loads": 0}

    def load_models(cfg, device):
        ctx["loads"] += 1
        return ("DETECTOR_STUB", "PIPELINE_STUB")

    def load_default_cfg():
        return {"model": {"device": "cuda:0"}}

    ns = {"threading": threading, "load_models": load_models,
          "load_default_cfg": load_default_cfg}
    exec(compile(class_src, f"<{class_name}>", "exec"), ns)
    Mgr = ns[class_name]

    barrier = threading.Barrier(workers)          # 让所有线程同时冲进 get_instance
    got = []

    def worker():
        barrier.wait()
        got.append(Mgr.get_instance({"model": {"device": "cuda:0"}}, "cuda:0"))

    threads = [threading.Thread(target=worker) for _ in range(workers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(got) == workers, f"只拿到 {len(got)}/{workers} 个结果"
    n_inst = len({id(i) for i in got})
    assert n_inst == 1, f"{workers} 线程并发得到 {n_inst} 个不同实例 (应 1)"
    assert ctx["loads"] == 1, f"load_models 被调用了 {ctx['loads']} 次 (应 1)"

    # 模拟后续请求: 再调任意次, 同一实例、不再加载
    again = Mgr.get_instance(None, None)          # cfg/device 传 None 应被忽略
    assert again is got[0], "再次 get_instance() 不是同一个实例"
    assert ctx["loads"] == 1, f"再次调用后 load_models 变成 {ctx['loads']} 次"

    print(f"  {PASS} {class_name}: {workers} 线程并发 get_instance "
          f"→ 实例数=1, load_models={ctx['loads']}, 再调仍同一实例")
    return ctx["loads"]


# ──────────────────────────────────────────────────────────── 启动即加载断言 ────
def check_infer_startup(repo):
    server_src = load_source(os.path.join(repo, "HSMR-infer", "hsmr_infer", "server.py"))
    tree = ast.parse(server_src)

    lif = find_function(tree, "lifespan")
    assert lif is not None, "server.py 找不到 lifespan"
    cp = first_call_pos(lif, "HSMRModelManager.get_instance")
    yp = first_yield_pos(lif)
    assert cp is not None, "lifespan 里没有 HSMRModelManager.get_instance() 调用"
    assert yp is not None, "lifespan 里没有 yield"
    assert cp < yp, f"get_instance 在 yield 之后 (第{cp[0]}行 > 第{yp[0]}行) → 不是启动加载"
    print(f"  {PASS} server.py lifespan: get_instance() 在第{cp[0]}行, yield 在第{yp[0]}行 → 启动即加载")

    infer_fn = find_function(tree, "infer")
    assert infer_fn is not None, "server.py 找不到 /infer 处理函数"
    assert not call_dotted_in(infer_fn, "load_models"), "/infer 里直接调了 load_models (会重复加载!)"
    assert has_state_runtime_subscript(infer_fn), "/infer 没有用 STATE['runtime'] 单例"
    print(f"  {PASS} /infer 处理函数: 无 load_models 调用, 复用 STATE['runtime'] 单例")
    return True


def check_render_startup(repo):
    main_src = load_source(os.path.join(repo, "HSMR-render", "main.py"))
    inf_src = load_source(os.path.join(repo, "HSMR-render", "inference.py"))
    mtree = ast.parse(main_src)

    # SKEL 渲染器双检锁
    get_skel = find_function(mtree, "_get_skel_renderer")
    assert get_skel is not None, "main.py 找不到 _get_skel_renderer"
    assert_double_checked_lock(get_skel, "_SKEL_RENDERER")
    print(f"  {PASS} main.py _get_skel_renderer: 双检锁结构完整 (if None → with LOCK → if None → 赋值)")

    # 渲染任务复用渲染器, 而非重建
    assert call_dotted_in(mtree, "_get_skel_renderer.render"), "没有 _get_skel_renderer().render(...) 复用"
    print(f"  {PASS} 渲染任务用 _get_skel_renderer().render(...) 复用渲染器")

    # lifespan: SKEL 渲染器在 yield 前加载
    lif = find_function(mtree, "lifespan")
    assert lif is not None, "main.py 找不到 lifespan"
    cp = first_call_pos(lif, "_get_skel_renderer")
    yp = first_yield_pos(lif)
    assert cp is not None and yp is not None and cp < yp
    print(f"  {PASS} main.py lifespan: _get_skel_renderer(cfg) 第{cp[0]}行在 yield 第{yp[0]}行前 → 启动即加载")

    # RenderModelManager 也在 worker.start() 里于启动阶段 get_instance
    itree = ast.parse(inf_src)
    start_fn = None
    for node in ast.walk(itree):
        if isinstance(node, ast.ClassDef) and node.name == "InferenceWorker":
            start_fn = find_function(node, "start")
    assert start_fn is not None and call_dotted_in(start_fn, "RenderModelManager.get_instance")
    print(f"  {PASS} render inference.py InferenceWorker.start: RenderModelManager.get_instance 启动阶段复用")
    return True


# ──────────────────────────────────────────────────────────── 黑盒 (remote) ────
def _http_get_json(url, timeout=10):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def multipart_post(url, fields):
    """fields: [(name, filename_or_None, content_type_or_None, bytes)]"""
    boundary = "----hsmr-singleton-test-%d" % os.getpid()
    body = bytearray()
    for name, fname, ctype, data in fields:
        body += f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"'.encode()
        if fname:
            body += f'; filename="{fname}"'.encode()
        body += b"\r\n"
        if ctype:
            body += f"Content-Type: {ctype}\r\n".encode()
        body += b"\r\n" + data + b"\r\n"
    body += f"--{boundary}--\r\n".encode()
    req = urllib.request.Request(
        url, data=bytes(body),
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read().decode("utf-8"))


def run_remote(args):
    base = f"http://{args.host}:{args.port}"

    # 1) 启动即加载: 首个动作就查 /health (还没发任何 /infer)
    health = _http_get_json(base + "/health", timeout=10)
    print(f"  /health → {json.dumps(health, ensure_ascii=False)}")
    assert health.get("status") == "ok", f"health.status={health.get('status')}"
    assert health.get("model_loaded") is True, "model_loaded 应为 true"
    print(f"  {PASS} 服务启动即加载: 未发 /infer 前 /health 即 model_loaded:true")

    # 2) 并发 /infer 不复载 (仅推理服务有 /infer; 渲染服务用 --skip-infer, 靠日志计数验证)
    if not args.skip_infer:
        png = base64.b64decode(TEST_PNG_B64)

        def do_infer(_):
            return multipart_post(base + "/infer", [
                ("image", "test.png", "image/png", png),
                ("max_instances", None, None, b"5"),
            ])

        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            results = list(ex.map(do_infer, range(args.workers)))
        for r in results:
            assert "persons" in r and "infer_ms" in r, f"响应缺字段: {list(r)[:8]}"
        print(f"  {PASS} 并发 {args.workers} 个 /infer 全部成功, "
              f"infer_ms={[r['infer_ms'] for r in results]}")
    else:
        print(f"  (已跳过并发 /infer: 渲染服务无此端点, 以 docker 日志加载计数为准)")

    # 3) 可选: ssh 数容器日志里的加载次数
    if args.ssh_cmd:
        markers = {
            "hsmr-infer":  ("[启动] 模型已加载", "[加载]"),
            "hsmr-render": ("[加载]", "[inference] 模型已加载", "SKEL 骨骼渲染器就绪"),
        }.get(args.container, ("[加载]", "[启动]"))
        for m in markers:
            cmd = f'{args.ssh_cmd} \'docker logs {args.container} 2>&1 | grep -cF "{m}"\''
            out = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            n = out.stdout.strip() or out.stderr.strip()
            print(f"  容器 {args.container} 日志含 {m!r}: {n} 行")
        print(f"  (预期: 单次加载=2 行 '[加载]…'+'[加载] 完成…'; 启动标记各 1 行)")

    print("\n黑盒测试全部通过")
    return True


# ──────────────────────────────────────────────────────────── 入口 ────
def main():
    ap = argparse.ArgumentParser(description="HSMR 模型单例 + 启动即加载测试")
    ap.add_argument("mode", choices=["unit", "remote"])
    ap.add_argument("--repo", default=DEFAULT_REPO, help="工程根 (含 HSMR-infer/ HSMR-render/)")
    ap.add_argument("--workers", type=int, default=16, help="并发线程数 (默认 16)")
    ap.add_argument("--host", default="192.168.217.100", help="remote: 服务地址")
    ap.add_argument("--port", type=int, default=8010, help="remote: 服务端口")
    ap.add_argument("--container", default="hsmr-infer", help="remote: 容器名 (数日志用)")
    ap.add_argument("--skip-infer", action="store_true",
                    help="remote: 跳过并发 /infer (渲染服务无此端点时用)")
    ap.add_argument("--ssh-cmd", default=None,
                    help='remote: 远程执行命令前缀, 如 "/tmp/hsmr_ssh.sh" 或 "ssh naviai@host"; '
                         '给了才做 docker 日志加载次数核对')
    args = ap.parse_args()

    repo = os.path.abspath(args.repo)
    infer_inf = os.path.join(repo, "HSMR-infer", "hsmr_infer", "inference.py")
    render_inf = os.path.join(repo, "HSMR-render", "inference.py")
    for p in (infer_inf, render_inf):
        if not os.path.exists(p):
            print(f"{FAIL} 找不到 {p} (用 --repo 指定工程根)")
            sys.exit(1)

    if args.mode == "unit":
        print("== 1) 单例模式 (并发 get_instance) ==")
        run_singleton_concurrent(infer_inf, "HSMRModelManager", args.workers)
        run_singleton_concurrent(render_inf, "RenderModelManager", args.workers)

        print("\n== 2) fastapi 启动即加载 (静态断言) ==")
        check_infer_startup(repo)
        check_render_startup(repo)
        print("\n全部通过 ✅")
    else:
        run_remote(args)


if __name__ == "__main__":
    main()
