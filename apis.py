import time
from typing import Dict, List, Optional

import requests

from req_utils import generate_signature


def get_registration_status(params: Dict[str, str]) -> str:
    """
    查询用户在大众云学平台上的注册状态
    
    Args:
        params (dict):
            params是一个dictionary，其中需要有"user_id_number"关键词，
            代表用户身份证号，或者管理员身份证号，或者单位名称，或者统一信用代码。如：{"user_id_number": "372323199509260348"}
    Returns:
        str: 用户在大众云学平台上的注册状态，如：
        
        '经查询，您在大众云学平台上的注册状态如下：
        姓名: 张三,
        状态: 已注册,
        注册时间: 2021-03-01,
        注册类型: 专技个人,
        管理员: 王芳芳,
        单位: 山东省济南市中心医院,'

        如果用户尚未注册，则返回：
        '经查询，您尚未在大众云学平台上注册'

        如果查询失败，则返回：
        '查询失败，请稍后再试'
    """
    url = "http://120.41.168.136:8600/customer/api/mgm/get-register-status-list"  
    secret_key = "123456"
    timestamp = str(int(time.time() * 1000))
    signature = generate_signature(timestamp, secret_key)
    # print(signature)

    params = {
        "secret": signature,
        "timestamp": timestamp,
        'platformId': '6003',
        "idNumber": str(params["user_id_number"])
    }

    response = requests.get(url, params=params)

    if response.status_code == 200:
        data = response.json()["data"]
        if len(data) == 0:
            return f"很抱歉，根据您提供的身份证号码{params['user_id_number']}，我没有找到任何注册信息，请确认您提供了正确的信息并重试"
        status = {
            "姓名": data[0]["real_name"] if "real_name" in data[0] else None,
            "单位": data[0]["du_name"] if "du_name" in data[0] else None,
            "身份证号": data[0]["account"] if "account" in data[0] else None,
            "审核状态": data[0]["status"] if "status" in data[0] else None,
            "注册类型": data[0]["roleType"] if "roleType" in data[0] else None,
            "审核机构": data[0]["unitName"] if "unitName" in data[0] else None,
            "机构类型"  : data[0]["unitType"] if "unitType" in data[0] else None,
            "注册时间": data[0]["register_time"] if "register_time" in data[0] else None,
            "超管姓名": data[0]["adminName"] if "adminName" in data[0] else None,
            "机构联系电话": data[0]["phone"] if "phone" in data[0] else None,
        }

        ret_str = [f"{k}: {v}" for k, v in status.items() if v is not None]
        ret_str = "  \n".join(ret_str)

        return f"经查询，您在大众云学平台上的注册状态如下：  \n{ret_str}"

    else:
        return "查询失败，请稍后再试"



def check_credit_hours(params: Dict[str, str]) -> List[dict]:
    """
    查询用户在大众云学平台上的学时情况

    Args:
        params (dict):
            params是一个dictionary，其中需要有"user_id_number", "year", "course_type"关键词，
                user_id_number: 代表用户身份证号，如："372323199509260348"
                year: 代表年份，如："2020"
                course_type: 代表课程类型，如："公需课"或者"专业课"
            如：{"user_id_number": "372323199509260348", "year": "2020", "course_type": "公需课"}
    Returns:
        List[dict]: 用户在大众云学平台上的学时情况，是一个list of dictionaries，如：
        [
            {"课程名称": "公需课5", "学时": 10, "进度": 100, "考核": "未完成"},
            {"课程名称": "公需课6", "学时": 10, "进度": 12, "考核": "未完成"},
        ]
    """
    pass

def check_course_purchases(params: Dict[str, str]) -> List[dict]:
    """
    查询用户在大众云学平台上的课程购买情况

    Args:
        params (dict):
            params是一个dictionary，其中需要有"user_id_number", "year", "course_name"关键词，
                user_id_number: 代表用户身份证号，如："372323199509260348"
                year: 代表年份，如："2020"
                course_name: 代表课程类型，如："新闻专业课培训班"
            如：{"user_id_number": "372323199509260348", "year": "2020", "course_name": "新闻专业课培训班"}
    Returns:
        dict: 用户在大众云学平台上的课程购买情况，如：
        {
            "课程名称": "新闻专业课培训班",
            "购买时间": "2020-03-01",
            "购买地点": "济南市",
            "管理员": "王芳芳",
            "学时": 10,
            "进度": 100,
            "考核": "合格",
        }
    """
    pass


### 一些模拟数据
## 注册信息
REGISTRATION_STATUS = {
    "372323199509260348": {
        "状态": "已注册",
        "注册时间": "2021-03-01",
        "注册地点": "济南市",
        "管理员": "王芳芳",
        "角色": "专技个人",
        "单位": "山东省济南市中心医院",
    },
}

## 课程购买信息
COURSE_PURCHASES = {
    "372323199509260348": {
        "2023": {
            "新闻专业课培训班": {
                "课程名称": "新闻专业课培训班",
                "课程类别": "专业课",
                "学时": 10,
                "进度": 90,
                "考核": "未完成",
                "购买时间": "2023-01-01",
                "购买地点": "山东省济南市",
                "培训机构": "山东省新闻学院",
            },
        },
        "2024": {
            "新闻专业课培训班": {
                "课程名称": "新闻专业课培训班",
                "课程类别": "专业课",
                "学时": 10,
                "进度": 0,
                "考核": "未完成",
                "购买时间": "2024-01-01",
                "购买地点": "山东省济南市",
                "培训机构": "山东省新闻学院",
            },
        },
    }
}

## 学时信息
CREDIT_HOURS = {
    "372323199509260348": {
        "2019": {
            "公需课": [
                {"课程名称": "公需课1", "学时": 10, "进度": 100, "考核": "合格"},
                {"课程名称": "公需课2", "学时": 10, "进度": 100, "考核": "合格"},
                {"课程名称": "公需课3", "学时": 10, "进度": 100, "考核": "未完成"},
                {"课程名称": "公需课4", "学时": 10, "进度": 85, "考核": "未完成"},
            ],
            "专业课": [
                {"课程名称": "专业课1", "学时": 10, "进度": 100, "考核": "合格"},
                {"课程名称": "专业课2", "学时": 10, "进度": 100, "考核": "合格"},
                {"课程名称": "专业课3", "学时": 10, "进度": 100, "考核": "未完成"},
                {"课程名称": "专业课4", "学时": 10, "进度": 85, "考核": "未完成"},
            ],
        },
        "2020": {
            "公需课": [
                {"课程名称": "公需课5", "学时": 10, "进度": 100, "考核": "未完成"},
                {"课程名称": "公需课6", "学时": 10, "进度": 12, "考核": "未完成"},
            ],
            "专业课": [
                {"课程名称": "专业课5", "学时": 10, "进度": 85, "考核": "未完成"},
            ],
        },
    }
}