import json
import os
import re
import sys
from typing import Any, List, Optional, Type

import apis
from langchain.tools import BaseTool
from statics import COURSE_PURCHASES


class RegistrationStatusToolIndividual(BaseTool):
    """查询专技个人在大众云学平台上的注册状态"""

    name: str = "专技个人注册状态查询工具"
    description: str = (
        "用于查询专技个人在大众云学平台上的注册状态，只有当用户明确提及需要帮助查询时调用，需要指通过 json 指定用户身份证号 user_id_number "
    )
    return_direct: bool = True

    def _run(self, params) -> Any:
        print(params)
        params_dict = params
        if "user_id_number" not in params_dict:
            return "抱歉，我还没有成功识别您的身份证号码，请指定"
        try:
            int(params_dict["user_id_number"])
        except Exception:
            return "抱歉，我还没有成功识别您的身份证号码，请指定"

        return apis.get_registration_status(params_dict)


class RegistrationStatusToolUniversal(BaseTool):
    """查询用户在大众云学平台上的注册状态"""

    name: str = "统一注册状态查询工具"
    description: str = (
        "用于查询用户在大众云学平台上的注册状态，只有当用户明确提及需要帮助查询时调用，需要指通过 json 指定查询号码 user_id_number "
    )
    return_direct: bool = True

    def _run(self, params) -> Any:
        print(params)
        params_dict = params
        if "user_id_number" not in params_dict:
            return "抱歉，我还没有成功识别您的身份证号码，单位信用代码，或者单位名称，请指定"
        try:
            int(params_dict["user_id_number"])
        except Exception:
            try:
                str(params_dict["user_id_number"])
            except Exception:
                return "抱歉，我还没有成功识别您的身份证号码，单位信用代码，或者单位名称，请指定"
        input = str(params_dict["user_id_number"])
        if input in ["unknown", "未知"]:
            return "抱歉，我还没有成功识别您的身份证号码，单位信用代码，或者单位名称，请指定"

        return apis.get_registration_status(params_dict)


class RegistrationStatusToolNonIndividual(BaseTool):
    """查询用人单位、主管部门或继续教育机构在大众云学平台上的注册状态"""

    name: str = "非个人注册状态查询工具"
    description: str = (
        "用于查询用人单位、主管部门或继续教育机构在大众云学平台上的注册状态，只有当用户明确提及需要帮助查询时调用，需要指通过 json 指定用户身份证号 user_id_number "
    )
    return_direct: bool = True

    def _run(self, params) -> Any:
        print(params)
        try:
            params_dict = json.loads(params)
        except json.JSONDecodeError:
            return "抱歉，我还没有成功识别您的单位管理员身份证号或者单位名称或者统一信用代码，请指定"
        if "user_id_number" not in params_dict:
            return "抱歉，我还没有成功识别您的单位管理员身份证号或者单位名称或者统一信用代码，请指定"
        try:
            str(params_dict["user_id_number"])
        except ValueError:
            return "抱歉，我还没有成功识别您的单位管理员身份证号或者单位名称或者统一信用代码，请指定"
        input = str(params_dict["user_id_number"])

        return apis.get_registration_status(params_dict)


class RegistrationStatusTool(BaseTool):
    """查询用户在大众云学平台上的注册状态"""

    name: str = "注册状态查询工具"
    description: str = (
        "用于查询用户在大众云学平台上的注册状态，需要指通过 json 指定用户身份证号 user_id_number "
    )
    return_direct: bool = True

    def _run(self, params) -> Any:
        print(params)
        try:
            params_dict = json.loads(params)
        except json.JSONDecodeError:
            return "请指定您或者管理员身份证号"
        if "user_id_number" not in params_dict:
            return "请指定您或者管理员身份证号"
        try:
            int(params_dict["user_id_number"])
        except ValueError:
            return "请指定您或者管理员身份证号"

        return apis.get_registration_status(params_dict)


class RefundTool(BaseTool):
    """根据用户回答，检查用户购买课程记录"""

    name: str = "检查用户购买课程记录工具"
    description: str = (
        "用于检查用户购买课程记录，需要指通过 json 指定用户身份证号 user_id_number、用户想要查询的课程年份 year、用户想要查询的课程名称 course_name "
    )
    # args_schema: Type[BaseModel] = CalculatorInput
    return_direct: bool = True

    def _run(self, params) -> Any:

        params = params.replace("'", '"')
        print(params, type(params))
        try:
            params_dict = json.loads(params)
        except json.JSONDecodeError as e:
            print(e)
            return "麻烦您提供一下您的身份证号，我这边帮您查一下"

        if "user_id_number" not in params_dict:
            return "麻烦您提供一下您的身份证号"
        if params_dict["user_id_number"] is None:
            return "麻烦您提供一下您的身份证号"
        if len(params_dict["user_id_number"]) < 2:
            return "麻烦您提供一下您正确的身份证号"

        if "year" not in params_dict:
            return "您问的是哪个年度的课程？如：2019年"
        if params_dict["year"] is None:
            return "您问的是哪个年度的课程？如：2019年"
        if len(params_dict["year"]) < 4:
            return "您问的是哪个年度的课程？如：2019年"

        if "course_name" not in params_dict:
            return "您问的课程名称是什么？如：新闻专业课培训班"
        if params_dict["course_name"] is None:
            return "您问的课程名称是什么？如：新闻专业课培训班"
        if len(params_dict["course_name"]) < 2:
            return "您问的课程名称是什么？如：新闻专业课培训班"

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
                        return "经查询您的这个课程没有学习，您可以点击右上方【我的学习】，选择【我的订单】，找到对应课程点击【申请售后】，费用在1个工作日会原路退回。"
                    return f"经查询，您的课程{course_name}学习进度为{progress}%，可以按照未学的比例退费，如需退费请联系平台的人工热线客服或者在线客服进行反馈。"
                return f"经查询，您在{year}年度，没有购买{course_name}，请您确认您的课程名称、年度、身份证号是否正确。"
            return f"经查询，您在{year}年度，没有购买{course_name}，请您确认您的课程名称、年度、身份证号是否正确。"
        return f"经查询，您在{year}年度，没有购买{course_name}，请您确认您的课程名称、年度、身份证号是否正确。"


class CheckPurchaseTool(BaseTool):
    """根据用户回答，检查用户购买课程记录"""

    name: str = "检查用户购买课程记录工具"
    description: str = (
        "用于检查用户购买课程记录，需要指通过 json 指定用户身份证号 user_id_number、用户想要查询的课程年份 year、用户想要查询的课程名称 course_name "
    )
    # args_schema: Type[BaseModel] = CalculatorInput
    return_direct: bool = True

    def _run(self, params) -> Any:

        params = params.replace("'", '"')
        print(params, type(params))
        try:
            params_dict = json.loads(params)
            params_dict = {k: str(v) for k, v in params_dict.items()}
        except json.JSONDecodeError as e:
            print(e)
            return "麻烦您提供一下您的身份证号，我这边帮您查一下"

        if "user_id_number" not in params_dict:
            return "麻烦您提供一下您的身份证号"
        if params_dict["user_id_number"] is None:
            return "麻烦您提供一下您的身份证号"
        if len(str(params_dict["user_id_number"])) < 2:
            return "麻烦您提供一下您正确的身份证号"

        if "year" not in params_dict:
            return "您问的是哪个年度的课程？如：2019年"
        if params_dict["year"] is None:
            return "您问的是哪个年度的课程？如：2019年"
        if len(str(params_dict["year"])) < 4:
            return "麻烦您确认你的课程年度。如：2019年"

        if "course_name" not in params_dict:
            return "您问的课程名称是什么？如：新闻专业课培训班"
        if params_dict["course_name"] is None:
            return "您问的课程名称是什么？如：新闻专业课培训班"
        if len(params_dict["course_name"]) < 2:
            return "请您提供您想要查询的课程的正确名称。如：新闻专业课培训班"

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
