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

        return apis.get_registration_status_api(params_dict)


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

        return apis.get_registration_status_api(params_dict)


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

        return apis.get_registration_status_api(params_dict)


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

        return apis.get_registration_status_api(params_dict)


class RefundTool(BaseTool):
    """根据用户回答，检查用户购买课程记录"""

    name: str = "检查用户购买课程记录工具"
    description: str = (
        "用于检查用户购买课程记录，需要指通过 json 指定用户身份证号 user_id_number、用户想要查询的课程年份 year、用户想要查询的课程名称 course_name "
    )
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

        return apis.check_course_refund_api(params_dict)


class CheckPurchaseTool(BaseTool):
    """根据用户回答，检查用户购买课程记录"""

    name: str = "检查用户购买课程记录工具"
    description: str = (
        "用于检查用户购买课程记录，需要指通过 json 指定用户身份证号 user_id_number、用户想要查询的课程年份 year、用户想要查询的课程名称 course_name "
    )
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

        return apis.check_course_purchases_api(params_dict)
