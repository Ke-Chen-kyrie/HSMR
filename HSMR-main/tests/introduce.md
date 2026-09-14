一、两个接口当前的返回值(先确定)
GET /render/queue

{ "queue_size": 2 }          // 当前渲染队列长度(0~50)
GET /render/pop —— 取一张,取走即移除

{
  "rendered": {
    "joints": ["hand_l"],            // 选中的关节(英文)
    "joints_cn": ["左手"],           // 中文
    "frame": "HEAD",                 // 坐标系
    "timestamp": 1786014517.5,       // 推理帧 unix 时间戳
    "position_m": {"hand_l": [0.246, 0.309, 1.992]},   // 关节位置
    "rotation_matrix": {"hand_l": [[...], ...]},        // 旋转矩阵
    "image_base64": "<jpg base64>"   // ← 前端显示要用的
  }
}
空队列返回 {"rendered": null}。
前端显示核心就是 image_base64 → data:image/jpeg;base64,...。
二、三种前端显示方案
方案 A — POST 直接返回 base64(不用队列接口)

POST /joint?include_base64=true&frame=HEAD
→ 响应.image_base64 → <img src="data:image/jpeg;base64,...">
✅ 一次请求拿图,天然是"最新一张",实现最简单
❌ 每次 POST 都带整图 base64(约几百 KB);队列没人 drain 会堆积
方案 B — 轮询 /render/queue + /render/pop(你指的两个接口)

1. 触发渲染:  POST /joint (body 选关节) —— 渲染入队,前端不需要 include_base64
2. 显示端循环:
   GET /render/queue → queue_size > 0 ?
   GET /render/pop   → 拿到 image_base64 → 显示
✅ 渲染与显示解耦;POST 响应体小
❌ pop 是消费即删除,显示端必须是唯一消费者,否则抢图;图片不落浏览器缓存;轮询有延迟
⚠️ 队列会堆积:_RENDER_QUEUE maxsize=50 且 put 会阻塞——没人 drain 的话,第 51 次 POST 的渲染会卡住直到 30s 超时。所以前端必须持续 drain 队列
方案 C(推荐)— 后端加静态 URL,图片直出

后端加一行:
  app.mount("/render/image", StaticFiles(directory=proj/data_outputs/depth3d/frames))
POST /joint 响应加: "rendered_image_url": "http://192.168.217.100:8002/render/image/hand_l.jpg"

前端:
  <img src="http://192.168.217.100:8002/render/image/hand_l.jpg">
✅ 浏览器原生加载/缓存/缩放,支持 <img onload>、缩略图、刷新自动生效
✅ POST 响应体小,队列完全不参与显示,可靠性最高
❌ 需小改后端(挂静态目录 + 返回 URL),前端拼接 host

三、我的建议
主推 方案 C,理由:显示"渲染后的图"本质是图片资源,最合理的形态就是 HTTP 静态 URL,而不是把 base64 传来传去。后端改动只有两处(约 5 行),前端就是普通 <img> 标签,还能白嫖浏览器缓存和懒加载。
如果不想改后端,就用 方案 B(你现在两个接口的路子),前端流程:

前端显示端(唯一消费者):
  let pending = true
  setInterval(600ms):
    q = await GET /render/queue
    while q.queue_size > 0:
      r = await GET /render/pop
      if r.rendered:
        img.src = 'data:image/jpeg;base64,' + r.rendered.image_base64
        caption = r.rendered.joints_cn.join('+') + ' @' + r.rendered.frame
注意三点:
唯一消费者:不要把 pop 给多个页面/进程用,会互相抢
必须 drain:轮询一定要把队列取到 0,否则 POST 渲染会阻塞超时
显示"最新" vs "逐张":pop 是 FIFO 逐张,如果只是显示最新一张,建议消费时一次 pop 全部直到 queue_size=0,只显示最后一张

四、(可选)如果允许微调后端,我推荐加这两个增强
增强
接口
为什么
GET /render/latest
peek 最新一张,不删除
显示"最新图"的场景最合适,不用抢队列、不破坏消费语义
/render/pop 返回加 image_url
配合方案 C
前端拿到 URL 直接 <img src>,不用处理 base64

要不要我把其中某个方案做成具体实施步骤(改哪个文件哪几行、前端伪代码完整版)?现在只到方案层面,不动代码。