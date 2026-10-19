"""Agent-authored semantic seeds + four controlled surface forms (not 300 independent teacher calls).

The seed bank is reviewable source. Labels and simulated returns are explicit,
not extracted from a teacher's unverifiable free-form answer.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from camera_sft.common import ROOT, write_json

LEX = {
 "camera":{"front":"前置镜头","rear":"后置镜头"},
 "format":{"jpeg":"JPEG格式","png":"PNG格式"},
 "quality":{"standard":"标准画质","high":"高画质"},
 "detail":{"connectivity":"连接状态","settings":"当前设置","health":"健康状态"},
 "target":{"person":"人员","vehicle":"车辆","pet":"宠物","package":"包裹","bicycle":"自行车"},
 "region":{"full":"整个画面","left":"画面左侧","right":"画面右侧"},
 "output":{"presence":"是否存在","boxes":"边界框位置"},
 "window":{"10m":"过去10分钟","1h":"过去1小时","24h":"过去24小时"},
 "limit":{"5":"至多5条","10":"至多10条","20":"至多20条"},
 "duration":{"10s":"10秒","30s":"30秒","60s":"60秒"},
 "resolution":{"720p":"720p分辨率","1080p":"1080p分辨率"},
 "audio":{"on":"开启声音","off":"关闭声音"},
 "profile":{"auto":"自动成像预设","daylight":"日光成像预设","night":"夜间成像预设","backlight":"逆光成像预设","low_motion":"低运动成像预设"},
 "exposure":{"low":"降低曝光补偿","neutral":"中性曝光补偿","high":"提高曝光补偿"},
 "mode_privacy":{"enabled":"启用隐私遮蔽","disabled":"禁用隐私遮蔽"},
 "mode_monitor":{"once":"单次监控","continuous":"持续监控"},
 "notification":{"on_first":"仅首次出现时通知","on_each_window":"每个窗口出现时通知"},
 "scope":{"current":"当前唯一监控任务","all":"所有监控任务"},
}
SPECS = {
 "camera_capture": [dict(camera=c,quality=q,format=f) for c,q,f in [('rear','high','png'),('front','standard','png'),('rear','standard','jpeg'),('front','high','jpeg'),('rear','high','jpeg')]],
 "camera_status": [dict(camera=c,detail=d) for c,d in [('rear','settings'),('front','health'),('rear','connectivity'),('front','settings'),('rear','health')]],
 "count_objects": [dict(camera=c,target=t,region=r) for c,t,r in [('rear','vehicle','right'),('front','pet','left'),('rear','person','left'),('front','vehicle','full'),('rear','pet','full')]],
 "detect_objects": [dict(camera=c,target=t,output=o) for c,t,o in [('rear','package','boxes'),('front','bicycle','presence'),('rear','pet','presence'),('front','vehicle','boxes'),('rear','person','boxes')]],
 "list_events": [dict(camera=c,target=t,window=w,limit=n) for c,t,w,n in [('rear','pet','24h','20'),('front','vehicle','10m','5'),('rear','person','1h','10'),('front','pet','1h','20'),('rear','vehicle','24h','5')]],
 "record_clip": [dict(camera=c,duration=d,resolution=r,audio=a) for c,d,r,a in [('rear','60s','1080p','off'),('front','10s','720p','on'),('rear','30s','720p','off'),('front','60s','1080p','on'),('rear','10s','1080p','on')]],
 "set_image_profile": [dict(camera=c,profile=p,exposure=e) for c,p,e in [('rear','backlight','high'),('front','low_motion','neutral'),('rear','daylight','low'),('front','auto','high'),('rear','night','neutral')]],
 "set_privacy": [dict(camera=c,mode=m) for c,m in [('rear','enabled'),('front','disabled'),('rear','enabled'),('rear','disabled'),('front','enabled')]],
 "start_monitor": [dict(camera=c,target=t,duration=d,mode=m,notification=n) for c,t,d,m,n in [('rear','vehicle','60s','continuous','on_each_window'),('front','pet','30s','once','on_first'),('rear','person','10s','continuous','on_first'),('front','vehicle','30s','continuous','on_each_window'),('rear','pet','60s','once','on_first')]],
 "stop_monitor": [dict(camera=c,scope=s) for c,s in [('rear','all'),('front','current'),('rear','all'),('rear','current'),('front','all')]],
}

def words(tool, args):
    return {k:LEX[('mode_privacy' if tool=='set_privacy' else 'mode_monitor') if k=='mode' else k][v] for k,v in args.items()}

def query(tool, w, index):
    c=w['camera']
    if tool=='camera_capture':return f"请让{c}保存一幅{w['quality']}的静态照片，文件采用{w['format']}。"
    if tool=='camera_status':return f"我想核对{c}的{w['detail']}，请读取这一项信息。"
    if tool=='count_objects':return f"请给出{c}当前{w['region']}中{w['target']}的数目。"
    if tool=='detect_objects':return f"对于{c}现在这一帧的{w['target']}，请返回{w['output']}。"
    if tool=='list_events':return f"请检索{c}在{w['window']}留下的{w['target']}事件记录，返回{w['limit']}。"
    if tool=='record_clip':return f"请用{c}录一段视频：总长度{w['duration']}，{w['resolution']}，{w['audio']}。"
    if tool=='set_image_profile':return f"请将{c}调整为{w['profile']}，同时设为{w['exposure']}，这是一次成像设置操作。"
    if tool=='set_privacy':return f"请对{c}{w['mode']}；本次只调整这一项隐私设置。" + ("不涉及停止监控。" if index==2 else "")
    if tool=='start_monitor':
        duration='每个检测窗口' if w['mode']=='持续监控' else '总运行时长'
        return f"请在{c}对{w['target']}启动{w['mode']}，{duration}{w['duration']}，{w['notification']}。"
    return f"请结束{c}的{w['scope']}，这次不操作录像。" + ("只处理给定任务。" if index==2 else "")

def outcome(tool, args, j):
    if j==4:return {'ok':False,'error':'device_unavailable'},'设备暂不可用，本次操作未完成。'
    if tool=='camera_capture':return {'ok':True,'photo_id':f'photo-{j+1}'},f"照片已保存，编号 photo-{j+1}。"
    if tool=='camera_status':return {'ok':True,'detail':args['detail'],'value':'normal'},'所查询的状态正常。'
    if tool=='count_objects':return {'ok':True,'count':j+2},f"当前指定区域内共有{j+2}个目标。"
    if tool=='detect_objects':
        if args['output']=='presence':return {'ok':True,'present':False},'当前画面中未检测到指定目标。'
        return {'ok':True,'boxes':[[10,20,50,80]]},'检测到目标，边界框为[10,20,50,80]。'
    if tool=='list_events':return {'ok':True,'events':[]},'指定时间范围内没有符合条件的历史事件。'
    if tool=='record_clip':return {'ok':True,'clip_id':f'clip-{j+1}'},f"录像已保存，编号 clip-{j+1}。"
    if tool=='set_image_profile':return {'ok':True,'profile':args['profile'],'exposure':args['exposure']},'成像预设与曝光补偿已按要求更新。'
    if tool=='set_privacy':return {'ok':True,'mode':args['mode']},'隐私遮蔽已开启。' if args['mode']=='enabled' else '隐私遮蔽已关闭。'
    if tool=='start_monitor':return {'ok':True,'task_id':f'task-{j+1}','state':'running'},f"监控任务 task-{j+1} 已启动。"
    return {'ok':True,'stopped_ids':['fixture-task']},'给定的监控任务已停止。'

NEG = {
 'no_tool':[
 ('仅作知识问答：请解释镜头焦距与视角的大致关系。','一般来说，焦距越短视角越宽，焦距越长视角越窄。'),
 ('这是翻译练习，请把“隐私保护”译成英语，不执行设备操作。','“隐私保护”可以译为 privacy protection。'),
 ('只做算术题：7乘以8等于多少？','7乘以8等于56。'),
 ('我正在写说明书，请用一句话解释视频帧率。','视频帧率表示每秒显示或记录的画面帧数。'),
 ('请把“请等待”改写得更礼貌，这不是摄像头操作请求。','请您稍候片刻。'),
 ('请按从小到大列出数字9、2、6。','2、6、9。'),
 ('请说明PNG是否支持透明通道，我只需要知识解答。','PNG支持透明通道。'),
 ('只做文本判断：“镜头”这个词一共有几个汉字？','“镜头”共有两个汉字。')],
 'unsupported':[
 ('请把后置镜头的照片以TIFF格式保存，画质选择标准。','当前拍照工具不支持TIFF格式，可选择JPEG或PNG。'),
 ('请将前置镜头的视频流直接推送到一个RTSP地址。','当前工具没有RTSP推流能力，无法执行该操作。'),
 ('请让后置镜头录制45秒视频，1080p，声音关闭。','当前录像时长仅支持10秒、30秒或60秒，无法直接录制45秒。'),
 ('请在前置镜头的画面中识别每个人的姓名。','当前工具无法识别人的姓名。'),
 ('请删除后置镜头昨天的全部录像文件。','当前工具没有删除录像文件的能力。'),
 ('请将前置镜头的焦距设置为35毫米。','当前工具没有焦距调节能力。'),
 ('请查询后置镜头过去7天的宠物事件，最多返回5条。','当前事件查询不支持过去7天，可选择10分钟、1小时或24小时。'),
 ('请用后置镜头统计当前整个画面里的椅子数量。','当前计数工具不支持椅子类别。')],
 'clarification_required':[
 ('请用后置镜头保存PNG照片。','请说明照片需要标准画质还是高画质。'),
 ('请读取前置镜头的状态。','请说明要查询连接状态、当前设置还是健康状态。'),
 ('请统计后置镜头这一帧的车辆数量。','请说明统计整个画面、左侧还是右侧。'),
 ('请用前置镜头录制30秒视频，采用1080p分辨率。','请说明录像是否开启声音。'),
 ('请把后置镜头设置成逆光成像预设。','请说明曝光补偿要降低、保持中性还是提高。'),
 ('请停止前置镜头当前的监控任务，暂时没有任务状态信息。','请先提供当前监控任务信息，以确认需要停止的唯一任务。')],
 'multi_action':[
 ('请用后置镜头拍高画质PNG照片，然后读取前置镜头的健康状态。','当前一次请求只处理一个独立操作，请拆分拍照和状态查询。'),
 ('请在后置镜头关闭隐私遮蔽，并用前置镜头录10秒720p有声视频。','当前一次请求只处理一个独立操作，请分别提出隐私设置和录像请求。'),
 ('请读取前置镜头的连接状态，再读取后置镜头的当前设置。','当前一次请求只处理一个独立操作，请分别提出两个状态查询。')],
}

seeds=[]
for tool,variants in SPECS.items():
    for j,args in enumerate(variants):
        w=words(tool,args); text=query(tool,w,j); result,answer=outcome(tool,args,j)
        previous=None
        if tool=='stop_monitor':
            previous={'tool_id':'start_monitor','status':'ok','active_monitors':[{'id':'fixture-task','camera':args['camera']}]}
        seeds.append({'seed_id':f'{tool}-{j+1:02d}','category':'tool_failure' if j==4 else 'supported',
          'split':'dev' if j==3 or (j==4 and tool=='record_clip') else 'train',
          'query':text,'tool':tool,'args':args,'evidence':w,'previous_tool_result':previous,'result':result,'answer':answer})
for cat,items in NEG.items():
    for j,(text,answer) in enumerate(items):
        seeds.append({'seed_id':f'{cat}-{j+1:02d}','category':cat,'split':'dev' if j==len(items)-1 else 'train',
          'query':text,'tool':None,'args':{},'evidence':{},'previous_tool_result':None,'result':None,'answer':answer})
assert len(seeds)==75
assert sum(s['split']=='dev' for s in seeds)==15
write_json(ROOT/'data/seeds/seed_bank.json',{'provenance':'Authored by Codex in this project; deterministic controlled expansion; no external teacher API calls.',
 'semantic_review':'Author-reviewed seed intentions only; independent sample-level semantic review pending.',
 'surface_forms':['{query}','本次请求如下：{query}','请严格按以下要求处理：{query}','我这次的具体需求是：{query}'],
 'lexicon':LEX,'seeds':seeds})
print('75 agent-authored seeds; 60 train / 15 dev; four controlled forms => 300 trajectories.')
