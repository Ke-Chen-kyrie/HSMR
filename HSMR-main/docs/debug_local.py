"""
本地 CPU 一步步 debug HSMR — 不用连机器人

用一张本地图片跑完整 HSMR 流程，每步打印中间结果，帮助理解。
注意: CPU 慢(~60秒/帧), 耐心等。

用法:
  source .venv/bin/activate
  python docs/debug_local.py --image data_inputs/demo/example_imgs/ballerina.png

输出:
  docs/_debug_2d_overlay.png   2D渲染(人框+网格)
  docs/_debug_3d.png           3D场景(相机+人体+朝向)
  docs/_debug_data.json        中间数据(每步结果)
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
import numpy as np, cv2
import matplotlib
matplotlib.use('Agg')
from matplotlib import font_manager
import matplotlib.pyplot as plt

for _f in ['/usr/share/fonts/fonts-gb/GB_ST_GB18030.ttf']:
    if os.path.exists(_f):
        try:
            font_manager.fontManager.addfont(_f)
            matplotlib.rcParams['font.sans-serif']=[font_manager.FontProperties(fname=_f).get_name(),'DejaVu Sans']
            matplotlib.rcParams['font.family']='sans-serif'
            matplotlib.rcParams['axes.unicode_minus']=False
            break
        except Exception: pass

from lib.modeling.pipelines.vitdet import build_detector
from lib.kits.hsmr_demo import IMG_MEAN_255, IMG_STD_255, _img_det2patches, prepare_mesh, visualize_full_img
from lib.body_models.skel_utils.transforms import params_q2rot
import torch

BONES = [[0,11],[11,12],[12,13],[0,1],[1,2],[2,3],[3,4],[4,5],[0,6],[6,7],[7,8],[8,9],[9,10],[12,14],[14,15],[15,16],[16,17],[17,18],[12,19],[19,20],[20,21],[21,22],[22,23]]
JNAMES = ['pelvis','femur-R','tibia-R','talus-R','calcn-R','toes-R','femur-L','tibia-L','talus-L','calcn-L','toes-L','lumbar','thorax','head','scapula-R','humerus-R','ulna-R','radius-R','hand-R','scapula-L','humerus-L','ulna-L','radius-L','hand-L']

def rot_to_quat(M):
    M=np.asarray(M,dtype=float); tr=M[0,0]+M[1,1]+M[2,2]
    if tr>0:
        s=np.sqrt(tr+1)*2; q=[0.25*s,(M[2,1]-M[1,2])/s,(M[0,2]-M[2,0])/s,(M[1,0]-M[0,1])/s]
    else:
        i=int(np.argmax([M[0,0],M[1,1],M[2,2]]))
        if i==0:
            s=np.sqrt(1+M[0,0]-M[1,1]-M[2,2])*2
            q=[(M[2,1]-M[1,2])/s,0.25*s,(M[0,1]+M[1,0])/s,(M[0,2]+M[2,0])/s]
        elif i==1:
            s=np.sqrt(1+M[1,1]-M[0,0]-M[2,2])*2
            q=[(M[0,2]-M[2,0])/s,(M[0,1]+M[1,0])/s,0.25*s,(M[1,2]+M[2,1])/s]
        else:
            s=np.sqrt(1+M[2,2]-M[0,0]-M[1,1])*2
            q=[(M[1,0]-M[0,1])/s,(M[0,2]+M[2,0])/s,(M[1,2]+M[2,1])/s,0.25*s]
    q=np.array(q)/np.linalg.norm(q); return q.tolist()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--image', default='data_inputs/demo/example_imgs/ballerina.png')
    ap.add_argument('--max_instances', type=int, default=2)
    ap.add_argument('--out', default='docs/_debug')
    args=ap.parse_args()

    t_all=time.time()
    print("="*55)
    print("本地 CPU 逐步 debug HSMR")
    print("="*55)

    # ── 步骤0: 加载图片 ──
    print(f"\n[0] 加载图片: {args.image}")
    frame_bgr=cv2.imread(args.image)
    frame_rgb=cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    print(f"    尺寸: {frame_rgb.shape}")

    # ── 步骤1: 加载模型(一次, 慢) ──
    print("\n[1] 加载 detector + HSMR (CPU, 较慢)...")
    t0=time.time()
    detector=build_detector(batch_size=1, max_img_size=512, device='cpu', use_amp=False)
    from lib.modeling.pipelines.hsmr import build_inference_pipeline
    pipeline=build_inference_pipeline(model_root='data_inputs/released_models/HSMR-ViTH-r1d1', device='cpu')
    print(f"    模型加载耗时 {time.time()-t0:.1f}s")

    # ── 步骤2: 检测 ──
    print("\n[2] 人体检测 (CPU ~26s)...")
    t0=time.time()
    det_out=detector([frame_rgb])
    d=det_out[0][0]
    print(f"    检测耗时 {time.time()-t0:.1f}s")
    print(f"    检测总数: {len(d['scores'])}")
    for i in range(len(d['scores'])):
        print(f"      #{i} class={d['pred_classes'][i].item()} score={d['scores'][i].item():.3f} box={[round(x,1) for x in d['pred_boxes'][i].numpy()]}")

    # ── 步骤3: 裁剪人 ──
    patches, bbx_cs=_img_det2patches(frame_rgb, d, det_out[1][0], args.max_instances)
    print(f"\n[3] 裁剪 {len(patches)} 人 patch: {patches.shape}")
    if len(patches)==0:
        print("    无人, 结束"); return
    for i,cs in enumerate(bbx_cs):
        print(f"      P{i} bbox(center,scale): [{cs[0]:.0f},{cs[1]:.0f},{cs[2]:.0f}]px")

    # ── 步骤4: HSMR ──
    print("\n[4] HSMR 推理 (CPU ~15s)...")
    t0=time.time()
    patches=patches.astype(np.float32)
    pn=(patches-IMG_MEAN_255)/IMG_STD_255
    pn=np.ascontiguousarray(pn.transpose(0,3,1,2))
    with torch.no_grad():
        outputs=pipeline(torch.from_numpy(pn))
    print(f"    HSMR 耗时 {time.time()-t0:.1f}s")
    pd_params={k:v.detach().cpu().clone() for k,v in outputs['pd_params'].items()}
    pd_cam_t=outputs['pd_cam_t'].detach().cpu().clone()
    print(f"    poses [{pd_params['poses'].shape}] (46姿态参数)")
    print(f"    betas [{pd_params['betas'].shape}] (10体型)")
    print(f"    camera_translation [{pd_cam_t.shape}]")
    for i in range(len(patches)):
        print(f"      P{i} camera_translation=[{pd_cam_t[i,0]:.2f},{pd_cam_t[i,1]:.2f},{pd_cam_t[i,2]:.1f}]  (tz=假深度)")

    # ── 步骤5: SKEL 关节 + 旋转矩阵 ──
    print("\n[5] SKEL 3D 关节 + 根旋转矩阵")
    m_skin,m_skel=prepare_mesh(pipeline, pd_params, include_skeleton=True, batch_size=1)
    skel_out=pipeline.skel_model(poses=pd_params['poses'].to('cpu'), betas=pd_params['betas'].to('cpu'), skelmesh=False)
    joints_body=skel_out.joints.detach().cpu().numpy()  # [N,44,3]
    rotations=params_q2rot(pd_params['poses'])  # [N,24,3,3]
    for i in range(len(patches)):
        rr=rotations[i,0].numpy()
        fwd=(rr @ np.array([0.0,0.0,1.0])).astype(float)
        print(f"      P{i} joints_44 [{joints_body[i].shape}]")
        print(f"          骨盆(8): {np.round(joints_body[i,8],3)}")
        print(f"          头(13):  {np.round(joints_body[i,13],3)}")
        print(f"          根旋转矩阵:\n{np.round(rr,3)}")
        print(f"          四元数(wxyz): {[round(x,3) for x in rot_to_quat(rr)]}")
        print(f"          前向: {np.round(fwd,3)}")

    # ── 步骤6: 渲染 ──
    print("\n[6] 渲染 2D 叠加 + 3D 场景 (CPU ~10s)")
    det_meta={'n_patch_per_img':[len(patches)],'bbx_cs_per_img':[bbx_cs],'bbx_cs':np.asarray(bbx_cs)}
    rendered, raw_cam_t=visualize_full_img(pd_cam_t, [frame_rgb], det_meta, m_skin, m_skel)
    cv2.imwrite(f'{args.out}_2d_overlay.png', cv2.cvtColor(rendered[0], cv2.COLOR_RGB2BGR))
    print(f"    → {args.out}_2d_overlay.png")

    # 3D 场景 (每人在虚拟相机系)
    fig=plt.figure(figsize=(14,7))
    ax=fig.add_subplot(111,projection='3d')
    ax.scatter([0],[0],[0],s=200,c='#a371f7',marker='s',edgecolors='white')
    ax.text(0,0.1,0.1,'相机',fontsize=10,color='#a371f7')
    for d,cc,n in [([1,0,0],'r','X'),([0,1,0],'g','Y'),([0,0,1],'b','Z')]:
        ax.quiver(0,0,0,*d,color=cc,linewidth=2,arrow_length_ratio=0.2)
    cols=['#f78166','#58a6ff','#3fb950']
    for i in range(len(patches)):
        jc=joints_body[i]+raw_cam_t[i]
        for a,b in BONES:
            ax.plot([jc[a,0],jc[b,0]],[jc[a,1],jc[b,1]],[jc[a,2],jc[b,2]],'-',color=cols[i%3],lw=2)
        ax.scatter(jc[:,0],jc[:,1],jc[:,2],c=cols[i%3],s=20)
        rr=rotations[i,0].numpy(); fwd=(rr@np.array([0.0,0.0,1.0])).astype(float)
        root=jc[0]
        ax.quiver(root[0],root[1],root[2],fwd[0]*0.8,fwd[1]*0.8,fwd[2]*0.8,color='#ffd166',linewidth=3,arrow_length_ratio=0.2)
        ax.text(root[0],root[1],root[2],f' P{i}',fontsize=10,color=cols[i%3])
    ax.set_xlabel('X');ax.set_ylabel('Y');ax.set_zlabel('Z')
    ax.set_title('3D 场景 (虚拟相机系: 相机原点, 人体在camera_translation处)',fontsize=12)
    ax.view_init(elev=15,azim=-60)
    plt.tight_layout(); plt.savefig(f'{args.out}_3d.png',dpi=120,bbox_inches='tight',facecolor='white');plt.close()
    print(f"    → {args.out}_3d.png")

    # ── 保存数据 ──
    data={'image':args.image,'persons':[]}
    for i in range(len(patches)):
        rr=rotations[i,0].numpy()
        data['persons'].append({
            'bbox_cs':[float(x) for x in bbx_cs[i]],
            'camera_translation':[float(x) for x in raw_cam_t[i]],
            'joints_44_body':[[float(x) for x in j] for j in joints_body[i]],
            'root_rotation':[[float(x) for x in row] for row in rr],
            'quaternion_wxyz':[float(x) for x in rot_to_quat(rr)],
            'forward':[float(x) for x in (rr@np.array([0.0,0.0,1.0])).astype(float)],
        })
    with open(f'{args.out}_data.json','w',encoding='utf-8') as f:
        json.dump(data,f,ensure_ascii=False,indent=1)
    print(f"    → {args.out}_data.json")

    print(f"\n总耗时 {time.time()-t_all:.1f}s (CPU)")
    print("完成! 打开 _debug_3d.png 和 _debug_2d_overlay.png 看结果")

if __name__=='__main__':
    main()
