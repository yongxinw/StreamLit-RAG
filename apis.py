import time
from typing import Dict, List, Optional

import requests

from req_utils import generate_signature
from utils import check_user_location
import re
from statics import COURSE_PURCHASES, CREDIT_HOURS, REGISTRATION_STATUS, URL


def get_registration_status_api(params: Dict[str, str]) -> str:
    """
    查询用户在大众云学平台上的注册状态

    Args:
        params (dict):
            params是一个dictionary，其中需要有"user_id_number"关键词，
            代表用户身份证号。测试使用：{"user_id_number": "150624196607198652"}
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
    url = f"http://{URL}/customer/api/mgm/get-register-status-list"
    secret_key = "123456"
    timestamp = str(int(time.time() * 1000))
    signature = generate_signature(timestamp, secret_key)

    params = {
        "secret": signature,
        "timestamp": timestamp,
        "platformId": "6002",
        "idNumber": str(params["user_id_number"]),
    }

    response = requests.get(url, params=params)

    if response.status_code == 200:
        data = response.json()["data"]
        if data is None:
            print("[ALERT] None Type returned from API")
            return f"很抱歉，根据您提供的身份证号码，我没有找到任何注册信息，请确认您提供了正确的信息并重试"
        if len(data) == 0:
            return f"很抱歉，根据您提供的身份证号码，我没有找到任何注册信息，请确认您提供了正确的信息并重试"
        status = {
            "姓名": data[0]["real_name"] if "real_name" in data[0] else None,
            "单位": data[0]["du_name"] if "du_name" in data[0] else None,
            "身份证号": data[0]["account"] if "account" in data[0] else None,
            "审核状态": data[0]["status"] if "status" in data[0] else None,
            "注册类型": data[0]["roleType"] if "roleType" in data[0] else None,
            "审核机构": data[0]["unitName"] if "unitName" in data[0] else None,
            "机构类型": data[0]["unitType"] if "unitType" in data[0] else None,
            "注册时间": (
                data[0]["register_time"] if "register_time" in data[0] else None
            ),
            "超管姓名": data[0]["adminName"] if "adminName" in data[0] else None,
            "机构联系电话": data[0]["phone"] if "phone" in data[0] else None,
        }

        ret_str = [f"{k}: {v}" for k, v in status.items() if v is not None]
        ret_str = "  \n".join(ret_str)

        return f"经查询，您在大众云学平台上的注册状态如下：  \n{ret_str}"

    else:
        return "查询失败，请稍后再试"


def check_credit_hours_simulate(
    params_dict: Dict[str, str], credit_problem_chain_executor
) -> str:
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
        str: 用户在大众云学平台上的学时情况
    """
    user_id_number = str(params_dict["user_id_number"])
    year = re.search(r"\d+", str(params_dict["year"])).group()
    course_type = str(params_dict["course_type"])

    template = credit_problem_chain_executor.agent.runnable.get_prompts()[
        0
    ].template.lower()
    # print(template)
    start_index = template.find("user location: ") + len("user location: ")
    end_index = template.find("\n", start_index)
    user_provided_loc = template[start_index:end_index].strip()

    user_loc = REGISTRATION_STATUS[user_id_number]["注册地点"]

    match_location = check_user_location(user_provided_loc, [user_loc])
    if match_location is None:
        match_other_loc = check_user_location(
            user_provided_loc,
            [
                "开放大学",
                "蟹壳云学",
                "专技知到",
                "文旅厅",
                "教师",
            ],
        )
        if match_other_loc is not None:
            if match_other_loc == "文旅厅":
                return "本平台只是接收方，学时如果和您实际不符，建议您先咨询您的学习培训平台，学时是否有正常推送过来，只有推送了我们才能收到，才会显示对应学时。"
            return f"经查询您本平台的单位所在区域是{user_loc}，不是省直，非省直单位学时无法对接。"
        return f"经查询您本平台的单位所在区域是{user_loc}，不是{user_provided_loc}，区域不符学时无法对接，建议您先进行“单位调转”,调转到您所在的地市后，再联系您的学习培训平台，推送学时。"
    else:
        match_other_loc = check_user_location(
            user_provided_loc,
            [
                "开放大学",
                "蟹壳云学",
                "专技知到",
                "文旅厅",
                "教师",
            ],
        )
        if match_other_loc is not None:
            return "请先咨询您具体的学习培训平台，学时是否有正常推送过来，只有推送了我们才能收到，才会显示对应学时。"
        hours = CREDIT_HOURS.get(user_id_number)
        if hours is None:
            return "经查询，平台还未接收到您的任何学时信息，建议您先咨询您的学习培训平台，学时是否全部推送，如果已确定有推送，请您24小时及时查看对接情况；每年7月至9月，因学时对接数据较大，此阶段建议1-3天及时关注。"
        year_hours = hours.get(year)
        if year_hours is None:
            return f"经查询，平台还未接收到您在{year}年度的任何学时信息，建议您先咨询您的学习培训平台，学时是否全部推送，如果已确定有推送，请您24小时及时查看对接情况；每年7月至9月，因学时对接数据较大，此阶段建议1-3天及时关注。"
        course_year_hours = year_hours.get(course_type)
        if course_year_hours is None or len(course_year_hours) == 0:
            return f"经查询，平台还未接收到您在{year}年度{course_type}的学时信息，建议您先咨询您的学习培训平台，学时是否全部推送，如果已确定有推送，请您24小时及时查看对接情况；每年7月至9月，因学时对接数据较大，此阶段建议1-3天及时关注。"
        total_hours = sum([x["学时"] for x in course_year_hours])
        finished_hours = sum(
            [
                x["学时"]
                for x in course_year_hours
                if x["进度"] == 100 and x["考核"] == "合格"
            ]
        )
        unfinished_courses = [
            f"{x['课程名称']}完成了{x['进度']}%"
            for x in course_year_hours
            if x["进度"] < 100
        ]
        untested_courses = [
            x["课程名称"] for x in course_year_hours if x["考核"] == "未完成"
        ]
        unfinished_str = "  \n\n".join(unfinished_courses)
        untested_str = "  \n\n".join(untested_courses)

        res_str = f"经查询，您在{year}年度{course_type}的学时情况如下：  \n\n"
        res_str += f"您报名的总学时：{total_hours}  \n\n"
        res_str += f"已完成学时：{finished_hours}  \n\n"
        res_str += f"其中，以下几节课进度还没有达到100%，每节课进度看到100%后才能计入学时  \n\n"
        res_str += unfinished_str + "  \n\n"
        res_str += f"以下几节课还没有完成考试，考试通过后才能计入学时  \n\n"
        res_str += untested_str + "  \n\n"
        return res_str


def check_credit_hours_api(
    params_dict: Dict[str, str], credit_problem_chain_executor
) -> str:
    """
    查询用户在大众云学平台上的学时情况

    Args:
        params (dict):
            params是一个dictionary，其中需要有"user_id_number", "year", "course_type"关键词，
                user_id_number: 代表用户身份证号，测试使用："350581199412080534"
                year: 代表年份，测试使用："2021"
                course_type: 代表课程类型，如："公需课"或者"专业课"
            如：{"泰安", "user_id_number": "350581199412080534", "year": "2021", "course_type": "公需课"}
    Returns:
        str: 用户在大众云学平台上的学时情况
    """
    user_id_number = str(params_dict["user_id_number"])
    year = re.search(r"\d+", str(params_dict["year"])).group()
    course_type = str(params_dict["course_type"])

    template = credit_problem_chain_executor.agent.runnable.get_prompts()[
        0
    ].template.lower()
    start_index = template.find("user location: ") + len("user location: ")
    end_index = template.find("\n", start_index)
    user_provided_loc = template[start_index:end_index].strip()

    user_loc = _get_user_location_by_id_number(user_id_number)

    match_location = check_user_location(user_provided_loc, [user_loc])
    match_other_loc = check_user_location(
        user_provided_loc,
        [
            "开放大学",
            "蟹壳云学",
            "专技知到",
            "文旅厅",
            "教师",
        ],
    )
    if match_location is None:
        if match_other_loc is not None:
            if match_other_loc == "文旅厅":
                return "本平台只是接收方，学时如果和您实际不符，建议您先咨询您的学习培训平台，学时是否有正常推送过来，只有推送了我们才能收到，才会显示对应学时。"
            return f"经查询您本平台的单位所在区域是{user_loc}，不是省直，非省直单位学时无法对接。"
        return f"经查询您本平台的单位所在区域是{user_loc}，不是{user_provided_loc}，区域不符学时无法对接，建议您先进行“单位调转”,调转到您所在的地市后，再联系您的学习培训平台，推送学时。"
    else:
        if match_other_loc is not None:
            return "请先咨询您具体的学习培训平台，学时是否有正常推送过来，只有推送了我们才能收到，才会显示对应学时。"
        ret_str = _get_credit_hours_by_id_number(user_id_number, year, course_type)
        return ret_str


def _get_user_location_by_id_number(user_id_number: str) -> Optional[str]:
    """
    根据用户身份证号获取用户所在地

    Args:
        user_id_number (str): 用户身份证号
    Returns:
        str: 用户所在地
    """
    url = f"http://{URL}/customer/api/mgm/get-professional-person-list"
    secret_key = "123456"
    timestamp = str(int(time.time() * 1000))
    signature = generate_signature(timestamp, secret_key)
    params = {
        "secret": signature,
        "timestamp": timestamp,
        "platformId": "6002",
        "idNumber": str(user_id_number),
    }
    response = requests.get(url, params=params)

    if response.status_code == 200:
        data = response.json()
        loc = data.get("data", {})[0].get("area", None)
        return loc

    return None


def _get_credit_hours_by_id_number(
    user_id_number: str, year: str, course_type: str
) -> Optional[str]:
    """
    根据用户身份证号获取用户学时信息

    Args:
        user_id_number (str): 用户身份证号
        year (str): 年份
        course_type (str): 课程类型
    Returns:
        str: 用户学时信息
    """
    credit_id = _get_credit_id_by_id_number(user_id_number)
    if credit_id is None:
        # Failed to retrieve credit id
        return "经查询，平台还未接收到您的任何学时信息，建议您先咨询您的学习培训平台，学时是否全部推送，如果已确定有推送，请您24小时及时查看对接情况；每年7月至9月，因学时对接数据较大，此阶段建议1-3天及时关注。"
    url = f"http://{URL}/customer/api/mgm/get-staff-report-by-id"
    secret_key = "123456"
    timestamp = str(int(time.time() * 1000))
    signature = generate_signature(timestamp, secret_key)

    params = {
        "secret": signature,
        "timestamp": timestamp,
        "platformId": "6002",
        "year": str(year),
        "id": str(credit_id),
    }

    response = requests.get(url, params=params)

    if response.status_code == 200:
        data = response.json()
        if data["totalCount"] == 0:
            return f"经查询，平台还未接收到您在{year}年度{course_type}的学时信息，建议您先咨询您的学习培训平台，学时是否全部推送，如果已确定有推送，请您24小时及时查看对接情况；每年7月至9月，因学时对接数据较大，此阶段建议1-3天及时关注。"
        data = data.get("data", {})
        if isinstance(data, list):
            data = data[0]
        else:
            return f"经查询，平台还未接收到您在{year}年度{course_type}的学时信息，建议您先咨询您的学习培训平台，学时是否全部推送，如果已确定有推送，请您24小时及时查看对接情况；每年7月至9月，因学时对接数据较大，此阶段建议1-3天及时关注。"
        if course_type == "公需课":
            needed_hours = data.get("year_publicNeeds", None)
            completed_hours = data.get("publicNeeds", None)
        elif course_type == "专业课":
            needed_hours = data.get("year_majorNeeds", None)
            completed_hours = data.get("majorNeeds", None)
        if any([needed_hours is None, completed_hours is None]):
            return "查询失败，请稍后再试，或者联系管理员或客服人员"
        if "pass_type" in data:
            if data["pass_type"] == 1:
                return f"经查询，您在{year}年度{course_type}的学时情况如下：\n\n您报名的总学时：{needed_hours}  \n\n已完成学时：{completed_hours}  \n\n您的认定结果：合格"
            else:
                return f"经查询，您在{year}年度{course_type}的学时情况如下：\n\n您报名的总学时：{needed_hours}  \n\n已完成学时：{completed_hours}  \n\n您的认定结果：不合格"
        return f"经查询，您在{year}年度{course_type}的学时情况如下：\n\n您报名的总学时：{needed_hours}  \n\n已完成学时：{completed_hours}"
    return "查询失败，请稍后再试，或者联系管理员或客服人员"


def _get_credit_id_by_id_number(user_id_number: str) -> Optional[str]:
    """
    根据用户身份证号获取用户学时信息

    Args:
        user_id_number (str): 用户身份证号
    Returns:
        dict: 用户学时信息
    """
    url = f"http://{URL}/customer/api/mgm/get-professional-person-list"
    secret_key = "123456"
    timestamp = str(int(time.time() * 1000))
    signature = generate_signature(timestamp, secret_key)
    params = {
        "secret": signature,
        "timestamp": timestamp,
        "platformId": "6002",
        "idNumber": str(user_id_number),
    }
    response = requests.get(url, params=params)

    if response.status_code == 200:
        data = response.json()
        credit_id = data.get("data", {})[0].get("id", None)
        return credit_id

    return None


def check_course_purchases_simulate(params_dict: Dict[str, str]) -> str:
    """
    查询用户在大众云学平台上的课程购买情况

    Args:
        params_dict (dict):
            user_id_number: 代表用户身份证号，如："372323199509260348"
            year: 代表年份，如："2020"
            course_name: 代表课程类型，如："新闻专业课培训班"
    Returns:
        str: 用户在大众云学平台上的课程购买情况
    """
    user_id_number = params_dict["user_id_number"]

    year = params_dict["year"]
    year = re.search(r"\d+", year).group()

    course_name = params_dict["course_name"]
    if COURSE_PURCHASES.get(user_id_number) is not None:
        purchases = COURSE_PURCHASES.get(user_id_number)
        if year in purchases:
            if course_name in purchases[year]:
                progress = purchases[year][course_name]["进度"]
                if progress == 0:
                    return f"经查询，您已经购买{year}年度的{course_name}，请前往专业课平台，点击右上方【我的学习】找到对应课程直接学习。"
                return f"经查询，您已经购买{year}年度的{course_name}，您的学习进度为{progress}%。请前往专业课平台，点击右上方【我的学习】找到对应课程继续学习。"
            return f"经查询，您在{year}年度，没有购买{course_name}，请您确认您的课程名称、年度、身份证号是否正确。"
        return f"经查询，您在{year}年度，没有购买{course_name}，请您确认您的课程名称、年度、身份证号是否正确。"
    return f"经查询，您在{year}年度，没有购买{course_name}，请您确认您的课程名称、年度、身份证号是否正确。"


def check_course_purchases_api(params_dict: Dict[str, str]) -> str:
    """
    查询用户在大众云学平台上的课程购买情况

    Args:
        params (dict):
            user_id_number: 代表用户身份证号，测试使用："ABCDE0670"
            year: 代表年份，测试使用："2023"
            course_name: 代表课程类型，测试使用："2023年_D_导入班级测试2"
    Returns:
        str: 用户在大众云学平台上的课程购买情况
    """
    user_id_number = params_dict["user_id_number"]

    year = params_dict["year"]
    year = re.search(r"\d+", year).group()

    course_name = params_dict["course_name"]

    url = f"http://{URL}/customer/api/train/get-general-info-list"
    secret_key = "123456"
    timestamp = str(int(time.time() * 1000))
    signature = generate_signature(timestamp, secret_key)

    params = {
        "secret": signature,
        "timestamp": timestamp,
        "platformId": "sit_001",
        "idNumber": user_id_number,
    }

    response = requests.get(url, params=params)

    if response.status_code == 200:
        data = response.json()
        data = data.get("data", {})
        if len(data) == 0:
            return f"经查询，您在{year}年度，没有购买任何课程，请您确认您的课程名称、年度、身份证号是否正确。"
        data = data[0]
        order_list = data.get("orderList", [])

        if len(order_list) == 0:
            return f"经查询，您在{year}年度，没有购买任何课程，请您确认您的课程名称、年度、身份证号是否正确。"
        order_list_in_year = [
            order for order in order_list if order["goodsDetail"][0]["year"] == year
        ]
        if len(order_list_in_year) == 0:
            return f"经查询，您在{year}年度，没有购买任何课程，请您确认您的课程名称、年度、身份证号是否正确。"

        matched_orders = [
            order
            for order in order_list_in_year
            if order["goodsDetail"][0]["goods_name"] == course_name
        ]
        if len(matched_orders) == 0:
            return f"经查询，您在{year}年度，没有购买{course_name}，请您确认您的课程名称、年度、身份证号是否正确。"

        matched_order = matched_orders[0]
        matched_course_id = matched_order["goodsDetail"][0]["goods_id"]

        class_list = data.get("classList", [])
        if len(class_list) == 0:
            return f"经查询，您已经购买{year}年度的{course_name}，请前往专业课平台，点击右上方【我的学习】找到对应课程直接学习。"
        class_list = class_list[0].get("classList", [])
        if len(class_list) == 0:
            return f"经查询，您已经购买{year}年度的{course_name}，请前往专业课平台，点击右上方【我的学习】找到对应课程直接学习。"

        matched_classes = [cls for cls in class_list if cls["id"] == matched_course_id]
        if len(matched_classes) == 0:
            return f"经查询，您已经购买{year}年度的{course_name}，请前往专业课平台，点击右上方【我的学习】找到对应课程直接学习。"

        progress = matched_classes[0]["progress"]
        return f"经查询，您已经购买{year}年度的{course_name}，您的学习进度为{progress}%。请前往专业课平台，点击右上方【我的学习】找到对应课程继续学习。"

    else:
        return "查询失败，请稍后再试，或者联系管理员或客服人员"


def check_course_refund_api(params_dict: Dict[str, str]) -> str:
    """
    查询用户在大众云学平台上的课程退费状态

    Args:
        params (dict):
            user_id_number: 代表用户身份证号，测试使用："ABCDE0670"
            year: 代表年份，测试使用："2023"
            course_name: 代表课程类型，测试使用："2023年_D_导入班级测试2"
    Returns:
        str: 用户在大众云学平台上的课程退费状态
    """
    user_id_number = params_dict["user_id_number"]

    year = params_dict["year"]
    year = re.search(r"\d+", year).group()

    course_name = params_dict["course_name"]

    url = f"http://{URL}/customer/api/train/get-general-info-list"
    secret_key = "123456"
    timestamp = str(int(time.time() * 1000))
    signature = generate_signature(timestamp, secret_key)

    params = {
        "secret": signature,
        "timestamp": timestamp,
        "platformId": "sit_001",
        "idNumber": user_id_number,
    }

    response = requests.get(url, params=params)

    if response.status_code == 200:
        data = response.json()
        data = data.get("data", {})
        if len(data) == 0:
            return f"经查询，您在{year}年度，没有购买任何课程，请您确认您的课程名称、年度、身份证号是否正确。"
        data = data[0]
        order_list = data.get("orderList", [])

        if len(order_list) == 0:
            return f"经查询，您在{year}年度，没有购买任何课程，请您确认您的课程名称、年度、身份证号是否正确。"
        order_list_in_year = [
            order for order in order_list if order["goodsDetail"][0]["year"] == year
        ]
        if len(order_list_in_year) == 0:
            return f"经查询，您在{year}年度，没有购买任何课程，请您确认您的课程名称、年度、身份证号是否正确。"

        matched_orders = [
            order
            for order in order_list_in_year
            if order["goodsDetail"][0]["goods_name"] == course_name
        ]
        if len(matched_orders) == 0:
            return f"经查询，您在{year}年度，没有购买{course_name}，请您确认您的课程名称、年度、身份证号是否正确。"

        matched_order = matched_orders[0]
        matched_course_id = matched_order["goodsDetail"][0]["goods_id"]

        class_list = data.get("classList", [])
        if len(class_list) == 0:
            return f"经查询，您已经购买{year}年度的{course_name}，请联系客服查询进度并按比例退费。"
        class_list = class_list[0].get("classList", [])
        if len(class_list) == 0:
            return f"经查询，您已经购买{year}年度的{course_name}，请联系客服查询进度并按比例退费。"

        matched_classes = [cls for cls in class_list if cls["id"] == matched_course_id]
        if len(matched_classes) == 0:
            return f"经查询，您已经购买{year}年度的{course_name}，请联系客服查询进度并按比例退费。"

        progress = matched_classes[0]["progress"]
        return f"经查询，您已经购买{year}年度的{course_name}，您的学习进度为{progress}%，可以按照未学的比例退费，如需退费请联系平台的人工热线客服或者在线客服进行反馈。"

    else:
        return "查询失败，请稍后再试，或者联系管理员或客服人员"


if __name__ == "__main__":
    user_id_number = "350581199412080534"
    # import ipdb
    # ipdb.set_trace()
    print(
        check_course_purchases_api(
            {
                "user_id_number": "ABCDE0670",
                "year": "2023",
                "course_name": "2023年_D_导入班级测试1",
            }
        )
    )
