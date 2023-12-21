import datetime
import json
import re

import cn2an

from utils import get_summary

def get_gender(text):  # 0女1男-1未知
    try:
        if re.match('^.*([，,、。]男[，,、。性]|性别男)', text):
            return 1
        if re.match('^.*([，,、。]女[，,、。性]|性别女)', text) or re.match('^.*女子监狱', text):
            return 0
        if re.match('^.*(强奸|妇女|幼女).{0,5}罪', text):
            return 1
        if re.match('^.{5,100}男.{0,20}生', text) and not re.match('^.{0,100}(罪犯|名).{0,2}男', text):
            return 1
        if re.match('^.{5,100}女.{0,20}生', text) and not re.match('^.{0,100}(罪犯|名).{0,2}女', text):
            return 0
        if not re.match('.*[男女]', text):
            return -1
        if re.match('^.*男', text):
            return 1
        else:
            return 0
    except:
        return -1


def get_nation(text):  # 0汉及未说明1少2外国港澳台3无国籍
    if re.match('^.{5,400}汉族', text):
        return 0
    elif re.match('^.{5,400}(国籍不明|无国籍)', text):
        return 3
    elif re.match('^.{5,400}(国籍|香港|澳门|台湾)', text):
        return 2
    elif re.match('^.{2,}族自', text) and not re.match('^.{5,}汉族', text):
        return 1
    elif re.match('^.{5,}[，,.。].{1,7}族[，,.。（]', text) and not re.match('^.{5,500}汉族', text):
        return 1
    else:
        return 0


def get_birth(text):  # 返回的是出生年份

    match_text = re.search("^.{5,200}\d{4}年\d{1,2}月\d{1,2}.{0,10}(生|出生)", text)
    if match_text is not None:
        birthdate = re.search("\d{4}年\d{1,2}月\d{1,2}", match_text.group())
        y = cn2an.cn2an(birthdate.group()[:4], mode="smart")
        return y if 1900 < y < 2010 else 0
    match_text = re.search("^.{5,200}(生于|出生).{0,5}\d{4}年\d{1,2}月\d{1,2}.{0,10}", text)
    if match_text is not None:
        birthdate = re.search('\d{4}年\d{1,2}月\d{1,2}', match_text.group())
        y = cn2an.cn2an(birthdate.group()[:4], mode="smart")
        return y if 1900 < y < 2010 else 0

    match_text = re.search('^.{5,200}[一二三四五六七八九零0〇]{4}年[一二三四五六七八九十]{1,2}月.{0,20}(生|出生)', text)
    if match_text is not None:
        birthdate = re.search('[一二三四五六七八九零0〇]{4}年[一二三四五六七八九十]{1,2}月', match_text.group())
        num_dict = "一二三四五六七八九"
        ystr = ""
        for token in birthdate.group()[:4]:
            if token == "零" or token == "0" or token == "〇":
                ystr += '0'
            else:
                ystr += str(num_dict.find(token) + 1)
        y = cn2an.cn2an(ystr, mode="smart")
        return y if 1900 < y < 2100 else 0

    return 0


def get_age(text, ym=str(datetime.datetime.now().year) + str(f"{datetime.datetime.now().month:02d}")):
    birth = int(get_birth(text))
    ym = int(ym[:4])
    return ym - birth if 0 < ym - birth < 100 else 0


def get_charge(text, charge_list):
    c_list = []
    match_list = re.findall('[以因犯].{1,20}?罪', text)
    match_list = list(set(match_list))
    if len(match_list) >= 1:
        # print(match_list)
        for m in match_list:
            if m[1:-1] in charge_list:
                c_list.append(m[1:-1])
        if len(c_list) != 0:
            return c_list
    match_list = re.findall('认定[\u4e00-\u9fa5]{0,10}犯[\u4e00-\u9fa5、]{1,20}罪', text)
    match_list = list(set(match_list))
    if len(match_list) >= 1:
        for m in match_list:
            mt = re.search("犯[\u4e00-\u9fa5、]{1,20}罪", m).group()
            if mt[1:-1] in charge_list:
                c_list.append(mt[1:-1])
        if len(c_list) != 0:
            return c_list
    match_list = re.findall('[以因][\u4e00-\u9fa5、]{1,30}罪.{0,5}判处', text)
    match_list = list(set(match_list))
    if len(match_list) >= 1:
        # print(match_list)
        for m in match_list:
            for c in charge_list:
                if c in m:
                    c_list.append(c)
        if len(c_list) != 0:
            return c_list

    match_list = re.findall('罪犯[\u4e00-\u9fa5]{0,10}因[\u4e00-\u9fa5、]{1,20}罪.{0,1}被', text)
    match_list = list(set(match_list))
    if len(match_list) >= 1:
        for m in match_list:
            for c in charge_list:
                if c in m:
                    c_list.append(c)
        if len(c_list) != 0:
            return c_list

    for c in charge_list:
        if c in text:
            c_list.append(c)
        if len(c_list) != 0:
            return c_list
    return c_list


def get_health(text):  # 0表示健康
    match_text = re.search('(患有.{0,20}疾病)|(糖尿病)|(高血压)|(身体.{0,4}病)|(患.{0,8}病)|(疾病)', text)
    if match_text is None:
        return 0
    else:
        return 1


def get_praise(text):
    match_text = re.search(
        "(表扬(奖励)?[0-9零一二三四五六七八九十]{1,3}[次个])|([0-9零一二三四五六七八九十]{1,3}[次个]表扬)",
        text)  # TODO 改
    if match_text is not None:
        text = match_text.group()
        times = re.sub("(表扬)|(奖励)|[次个]", '', text)
        return cn2an.cn2an(times, mode='smart')
    return text.count("表扬")


def split_apl(text):
    text = "".join(text.split())
    match_text = re.search('(依照|根据|依据).{0,20}(《.{2,50}》|中(国|华人民共和).{0,20}法》).{0,50}第.{0,10}条.*$',
                           text)
    if match_text is not None:
        fact_match = re.sub('(依照|根据|依据).{0,20}(《.{2,50}》|中(国|华人民共和).{0,20}法》).{0,50}第.{0,10}条.*$',
                            '', text)
        res_match = match_text.group()
        fact_match = re.sub('(为此[,，。]|据此[,，。]).{0,10}$', '', fact_match)
        return fact_match, res_match
    else:
        match_text = re.search(
            "(特(提请|报请)|监狱.{1,10}(意见|建议)|。.{0,40}建议.{0,20}(减去|减刑|暂予监外执行|假释)).*$",
            text)
        if match_text is not None:
            fact_match = re.sub(
                '(特(提请|报请)|监狱.{1,10}(意见|建议)|。.{0,40}建议.{0,20}(减去|减刑|暂予监外执行|假释)).*$', '',
                text)
            res_match = match_text.group()
            fact_match = re.sub('(为此[,，。]|据此[,，。]).{0,10}$', '', fact_match)
            return fact_match, res_match
        else:
            res_match = "。".join((text + "bar").split("。")[-3:-1])
            fact_match = "。".join((text + "bar").split("。")[:-3])
            fact_match = re.sub('(为此[,，。]|据此[,，。]).{0,10}$', '', fact_match)
            return fact_match, res_match


def split_jdg(text):
    text = "".join(text.split())
    match_text = re.search('本院认为.{2,30}(期间|表现|符合|减.{0,2}刑|假释|暂予监外执行).*', text)
    
    if match_text is not None:
        fact_match = re.sub('本院认为.{2,30}(期间|表现|符合|减.{0,2}刑|假释|暂予监外执行).*', '', text)
        res_match = re.sub("审判长.*$", '', match_text.group())
        return fact_match, res_match
    else:
        match_text = re.search('(依照|根据|依据).{0,20}(《.{2,50}》|中(国|华人民共和).{0,20}法》).{0,50}第.{0,10}条.*$',
                               text)
        if match_text is not None:
            fact_match = re.sub('(依照|根据|依据).{0,20}(《.{2,50}》|中(国|华人民共和).{0,20}法》).{0,50}第.{0,10}条.*$',
                                '', text)
            res_match = re.sub("审判长.*$", '', match_text.group())
            return fact_match, res_match
        else:
            fact_match = "。".join((text + "bar").split("。")[:-3])
            res_match = "。".join((text + "bar").split("。")[-3:-1])
            return fact_match, res_match


# 传入参数两个
def split(applyment=None, judgement=None):
    splited_text = {"fact_apl": None, "apl": None, "fact_jdg": None, "jdg": None}
    if applyment is not None:
        fact_apl, apl = split_apl(applyment)
        splited_text['fact_apl'] = fact_apl
        splited_text['apl'] = apl
    if judgement is not None:
        fact_jdg, jdg = split_jdg(judgement)
        splited_text['fact_jdg'] = fact_jdg
        splited_text['jdg'] = jdg

    return splited_text


def get_result(result, ym=str(datetime.datetime.now().year) + str(f"{datetime.datetime.now().month:02d}")):  # 驳回为0
    result = "".join(result.split())
    # 是否准许
    approval = 0
    match_text = re.search('(不符合|不予|驳回|不准|撤回|不给予).{0,6}(减刑|假释|暂予监外执行)', result)
    if match_text is not None:
        approval = 0
    match_text = re.search(
        '(减去|(准予|予以|给予)(假释|减.{0,2}刑)|减为|减余刑|减刑.{1,6}[月年]|去有期徒刑|暂予监外执行)', result)
    if match_text is not None:
        approval = 1
    if approval == 0:
        return 0
    ####
    # 准许减刑的

    match_text = re.search('减为(有期徒刑|有期.{0,4})(二十|20|十)', result)
    if match_text is not None:
        return 37
    match_text = re.search('为无期徒刑', result)
    if match_text is not None:
        return 38
    # match_text = re.search('(减.{0,10}([余残]).{0,10}刑|未执行完|尚未执行)', item['result'])
    # if match_text is not None:
    #     return 39
    match_text = re.search(
        '(减去有期徒刑[一二三四五六七八九十0-9]{1,3}年)|(减.{0,4}刑[一二三四五六七八九十0-9]{1,3}年)|(刑.{0,4}减((.{0,4}有期.{0,4})|((有期){0,1}))[一二三四五六七八九十0-9]{1,3}.{0,2}年)',
        result)
    reduction = 0
    if match_text is not None:
        match_text = re.search('[一二三四五六七八九十0-9]{1,3}', match_text.group())
        tnum = match_text.group()
        tnum = cn2an.cn2an(tnum, "smart")
        reduction = int(tnum * 12)
        match_text = re.search(
            '(减去有期徒刑[一二三四五六七八九十0-9]{1,3}年.{0,2}[一二三四五六七八九十0-9]{1,3}.{0,2}[个月])|(减.{0,4}刑[一二三四五六七八九十0-9]{1,3}年.{0,2}[一二三四五六七八九十0-9]{1,3}.{0,2}([个月]))|(刑.{0,4}减((.{0,4}有期.{0,4})|((有期){0,1}))[一二三四五六七八九十0-9]{1,3}.{0,2}年.{0,2}[一二三四五六七八九十0-9]{1,3}.{0,2}[个月])',
            result)
        if match_text is not None:
            text = re.sub(
                '(减去有期徒刑[一二三四五六七八九十0-9]{1,3}年)|(减.{0,4}刑[一二三四五六七八九十0-9]{1,3}年)|(刑.{0,4}减((.{0,4}有期.{0,4})|((有期){0,1}))[一二三四五六七八九十0-9]{1,3}.{0,2}年)',
                '', match_text.group())
            match_text = re.search('[一二三四五六七八九十0-9]{1,3}', text)
            tnum = match_text.group()
            tnum = cn2an.cn2an(tnum, "smart")
            reduction += int(tnum)
        return reduction
    match_text = re.search(
        '(减去有期徒刑[一二三四五六七八九十0-9]{1,4}.{0,2}月)|(减.{0,4}刑.{0,4}[一二三四五六七八九十0-9]{1,4}.{0,1}[个月])|(刑.{0,4}减((.{0,4}有期.{0,4})|((有期){0,1}))[一二三四五六七八九十0-9]{1,3}.{0,2}[个月])|(减.{0,4}有期.{0,4}[一二三四五六七八九十0-9]{1,4}.{0,2}[个月])',
        result)
    if match_text is not None:
        match_text = re.search('[一二三四五六七八九十0-9]{1,4}', match_text.group())
        tnum = match_text.group()
        tnum = cn2an.cn2an(tnum, "smart")
        return int(tnum)

    # 准许假释的

    result = re.sub('[○O]', '零', result)
    match_text = re.search('[自从即].{0,2}[〇零一二三四五六七八九十0-9]{3,4}年[一二三四五六七八九十0-9]{1,2}月.{1,10}([至到]){0,2}[〇零一二三四五六七八九十0-9]{3,4}年[一二三四五六七八九十0-9]{1,2}月',
                           result)
    if match_text is not None:
        stext = re.search('[自从即].{0,2}[〇零一二三四五六七八九十0-9]{3,4}年[一二三四五六七八九十0-9]{1,2}月',
                          match_text.group()[:-6]).group()
        etext = re.search('([至到]){0,2}[〇零一二三四五六七八九十0-9]{3,4}年[一二三四五六七八九十0-9]{1,2}月',
                          match_text.group()[-8:]).group()
        ytm = cn2an.cn2an(re.search('[〇零一二三四五六七八九十0-9]{3,4}', etext).group(), "smart") - cn2an.cn2an(
            re.search('[〇零一二三四五六七八九十0-9]{3,4}', stext).group(), "smart")

        mtm = cn2an.cn2an(re.search('[一二三四五六七八九十0-9]{1,2}', etext[-3:]).group(), "smart") - cn2an.cn2an(
            re.search('[一二三四五六七八九十0-9]{1,2}', stext[-3:]).group(), "smart")
        reduction = int(ytm * 12 + mtm)
        return reduction if 0 < reduction < 36 else 0

    match_text = re.search('([至到]){0,2}[〇零一二三四五六七八九十0-9]{3,4}年[一二三四五六七八九十0-9]{1,2}月',
                           result)
    if match_text is not None:
        etext = re.search('([至到]){0,2}[〇零一二三四五六七八九十0-9]{3,4}年[一二三四五六七八九十0-9]{1,2}月',
                          match_text.group()).group()
        ytm = cn2an.cn2an(re.search('[〇零一二三四五六七八九十0-9]{3,4}', etext).group(), "smart") - cn2an.cn2an(
            str(ym)[:4], "smart")
        # mtxt = etext[-3]
        # mtxtg = re.search('[一二三四五六七八九十0-9]{1,2}', mtxt)
        # if mtxtg is not None:
        mtm = cn2an.cn2an(re.search('[一二三四五六七八九十0-9]{1,2}', etext[-3:]).group(), "smart") - cn2an.cn2an(
            str(ym)[-2:], "smart")
        reduction = int(ytm * 12 + mtm)
        return reduction if 0 < reduction < 36 else 0
    return 0


def extract_info(applyment=None, judgement=None, ym=int(str(datetime.datetime.now().year) + str(datetime.datetime.now().month)), use_jdg=False):
        # TODO
        # 这部分用规则的方式抽取信息，返回的内容包括fact_text，之后整合进来
        fact_apl, apl, fact_jdg, jdg = None, None, None, None
        redu = -1
        if applyment is not None:  # 可先不弄
            fact_apl, apl = split_apl(applyment)
        if judgement is not None:
            fact_jdg, jdg = split_jdg(judgement)
            
            # 去除末尾不完整的
            pattern = re.compile(f'[.。][^.。]*$')
            fact_jdg = re.sub(pattern, '。', fact_jdg)
            
            redu = get_result(jdg, ym)
        if use_jdg:
            fact = fact_apl
        else:
            fact = fact_jdg
        gender = get_gender(fact)
        nation = get_nation(fact)
        age = get_age(fact)
        charge_list = json.load(open("assets/charges.json", 'rb'))
        charge = get_charge(fact, charge_list=charge_list)
        health = get_health(fact)
        praise = int(get_praise(fact))
        # redu 0为不允许，1-36为月份，37为无期减有期，38为死缓减无期
        ret = {
            "fact_apl": fact_apl,
            "apl": apl,
            "fact_jdg": fact_jdg,
            "jdg": jdg,
            "redu": redu,
            "gender": gender,
            "nation": nation,
            "age": age,
            "charge": charge,
            "health": health,
            "praise": praise
        }
        return ret



if __name__ == "__main__":

    print(get_result("本院认为，罪犯胡强强在服刑改造期间，认罪服法，认真遵守监规，接受教育改造，确有悔改表现，可以假释。依照《中华人民共和国刑法》第八十一条第一款、第八十三条及《中华人民共和国刑事诉讼法》第二百六十二条第二款之规定，裁定如下：对罪犯胡强强予以假释（假释考验期限自2015年6月25日起至2017年11月18日止）。本裁定送达后即发生法律效力。", 201506))