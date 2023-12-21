# 实现输入为整个文书，然后处理后接入model_shell，接受model_shell的输出，处理后给出输出
import time
import pickle
import torch
from model import ModelShell
from extract_info import extract_info


# 自定义参数类
class MyArgs:
    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = "assets/model"
        with open("assets/charge2id.pkl", "rb") as c2id:
            charge2id = pickle.load(c2id, encoding="utf8")
        self.charge_num = len(charge2id)
        self.class_num = 39
        self.ckpt_path = "assets/model/ckpt.bin"
        self.max_length = 512
    

# 速览切分后的结果
def quicklook(judgement):
    info = extract_info(applyment=None, judgement=judgement)
    return info['fact_jdg'], info['jdg']

def analyze(shell:ModelShell, judgement):
    info = extract_info(applyment=None, judgement=judgement)
    output = shell.get_output(info)
    output['fact_jdg'] = info['fact_jdg']
    output['jdg'] = info['jdg']
    conclusion = ""
    
    conclusion += f"罪犯民族：{'汉族' if output['nation'] == 0 else '少数民族' if output['nation'] == 1 else '其他'}, \n"
    conclusion += f"罪犯性别：{'男' if output['gender'] == 1 else '女'}, \n"
    conclusion += f"罪犯年龄：{'未知' if (output['age'] == 0 or output['age'] == 99) else output['age']}, \n"
    conclusion += f"罪犯罪名：{'未知' if len(output['charge']) == 0 else','.join(output['charge'])}, \n"
    conclusion += f"罪犯身体状况：{'健康' if output['health'] == 0 else '患病'}, \n"
    conclusion += f"罪犯在狱表现：受表扬{output['praise']}次, \n\n" if output['praise'] > 0 else '\n' 
    
    conclusion += f"建议减刑/假释：{output['suggestion']}个月\n"
    conclusion += f"判决偏差：{output['distance']}\n"
    conclusion += f"不确定度：{output['inconsistency']}\n"
    
    output["conclusion"] = conclusion
    
    return output


if __name__ == "__main__":
    
    args = MyArgs()
    shell = ModelShell(args)
    # 以上复用
    
    t1 = time.time()
    judgement = "河南省洛阳市中级人民法院刑事裁定书2016豫03刑更2432号罪犯师利利，男，1987年7月25日出生，汉族，河南省新安县人，初中文化程度，现在河南省第四监狱服刑。洛阳市中级人民法院于2011年11月15日作出2011洛刑一初字53号刑事判决书，认定被告人师利利犯故意伤害罪，判处有期徒刑十四年，剥夺政治权利三年。判决发生法律效力后即交付执行。在服刑期间，曾被减刑一次，减刑一年六个月。刑罚执行机关河南省第四监狱于2016年6月17日提出减刑建议书，报送本院审理。本院依法组成合议庭进行了审理。现已审理终结。刑罚执行机关河南省第四监狱经分监区集体评议、监区长办公会审核后公示二日、刑罚执行科审查、监狱提请减刑假释评审委员会评审后公示七日、监狱长办公会决定，并书面通报和邀请驻狱检察人员现场监督评审委员会评审活动等程序提出，该名罪犯确有悔改表现，并提供相关证据予以证实，建议对其减刑。经审理查明：原审认定2011年4月5日晚，被告人师利利与被害人柳XX等人一起到吃饭、唱歌，玩到次日凌晨1时30分，由贾X开车送众人回家，" + \
                "被告人师利与受害人柳XX发生口角、厮打，师利利用匕首朝柳XX胸部连刺两刀，柳XX经抢救无效死亡。罪犯师利利在服刑期间，能够认罪悔罪；自觉遵守监规狱纪，接受教育改造；积极参加思想、职业技术教育；积极参加劳动，完成生产任务。至本次提请减刑确定的考核截止日期2016年3月31日，共受表扬5次，被评为监狱级改造积极分子1次。刑罚执行机关提请对该名罪犯减刑，确已经过分监区集体评议、监区长办公会审核后公示二日、刑罚执行科审查、监狱提请减刑假释评审委员会评审后公示七日、监狱长办公会决定，并书面通报和邀请驻狱检察人员现场监督评审委员会评审活动等程序。上述事实有执行机关提供的生效判决书、执行通知书、减刑裁定书、罪犯计分考核情况汇总表、罪犯改造评审鉴定表、罪犯奖励审批表、罪犯改造积极分子审批表、罪犯减刑审核表、关于提请减刑经过程序的证明等证据在案佐证。本院认为，罪犯师利利自上次减刑以来确有悔改表现，符合减刑条件，可予减刑。根据其改造表现和所犯罪行及情节，依照《中华人民共和国刑事诉讼法》第二百六十二条第二款、《中华人民共和国刑法》第七十八条、第七十九条、《最高人民法院关于办理减刑、假释案件具体应用法律若干问题的规定》之规定，裁定如下：对罪犯师利利减去有期徒刑一年，剥夺政治权利刑期不变（有期徒刑的刑期从本裁定减刑之日起计算，即自2011年4月6日起至2022年10月5日止。本裁定送达后即发生法律效力"
    ql = quicklook(judgement=judgement)
    result = analyze(shell, judgement=judgement)
    print(args.device)
    print(result)
    t2 = time.time()
    print(t2 - t1)
    
