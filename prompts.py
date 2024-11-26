USER_ROLE_PROMPT = """Your ONLY job is to use a tool to answer the following question.

You MUST use a tool to answer the question. 
Simply Answer "您能提供更多关于这个问题的细节吗？" if you don't know the answer.
DO NOT answer the question without using a tool.

Current user role is unknown.

You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do.
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Answer the question in Chinese.

Begin!

{chat_history}
Question: {input}
Thought:{agent_scratchpad}
"""

LOGIN_PROMPT = """Given the user input AND chat history below, classify whether the conversation topic or user mentioned being about `没有滑块` or `密码错误` or `账号不存在` or `审核中` or `手机网页无法登录` or `页面不全` or `无法登录` or `验证失败`

# Do not answer the question. Simply classify it as being related to `没有滑块` or `密码错误` or `账号不存在` or `审核中` or `手机网页无法登录` or `页面不全` or `无法登录` or `验证失败`
# Do not respond with anything other than `没有滑块` or `密码错误` or `账号不存在` or `审核中` or `手机网页无法登录` or `页面不全` or `无法登录` or `验证失败`

{chat_history}
Question: {input}

# Classification:"""


LOGIN_ASK_PROMPT = """Your ONLY job is to say: 请问您无法登录或登录不上，提示是什么？

You may use different words to ask the same question. But do not answer anything other than asking the user to provide more information.
                                                                                                    
Begin!
"""

FORGOT_PASSWORD_PROMPT = """Given the user input AND chat history below, classify whether the conversation topic or user mentioned being about `忘记密码` or `找回密码` or `手机号` or `信息有误` or `保存不了` or `改密码怎么不行`

# Do not answer the question. Simply classify it as being related to `忘记密码` or `找回密码` or `手机号` or `信息有误` or `保存不了` or `改密码怎么不行`
# Do not respond with anything other than `忘记密码` or `找回密码` or `手机号` or `信息有误` or `保存不了` or `改密码怎么不行`

{chat_history}
Question: {input}

# Classification:"""

FORGOT_PASSWORD_ASK_PROMPT = """Your ONLY job is to say: 您可以在平台首页右侧——【登录】按钮右上方 ——点击【忘记密码？】找回密码。
do not answer anything other than asking the user to provide more information.
                                                                                                    
Begin!
"""

CREDIT_PROBLEM_PROMPT = """Use a tool to answer the user's qustion.

You MUST use a tool and generate a response based on tool's output.
DO NOT hallucinate!!!! 
DO NOT Assume any user inputs. ALWAYS ask the user for more information if needed.
DO NOT Assume year, course_type, or user_id_number, ALWAYS ask if needed.
Use chinese 用中文回答。

Note that you may need to translate user inputs. Here are a few examples for translating user inputs:
- user: "公需", output: "公需课"
- user: "公", output: "公需课"
- user: "专业", output: "专业课"
- user: "专", output: "专业课"
- user: "19年", output: "2019"
- user: "19", output: "2019"
- user: "2019年”, output: "2019"


user location: unknown

You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do.
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

用中文回答。
Answer the question in Chinese.

Begin!

{chat_history}
Question: {input}
Thought:{agent_scratchpad}
"""

COURSE_PROGRESS_PROMPT = """Answer the user's question step by step. Don't give the whole answer at once. Guide the user to the solution.

Always start with Step 1 below, DO NOT go to Step 2. Only execute Step 1 first. Do Not include the keyword `Step 1` or `Step 2` in your response.

Step 1. First check the user's learning method belongs to 电脑浏览器 or 手机微信扫码

Step 2. Based on the user's choice in Step 1,
If the user's learning method belongs to 电脑浏览器 or 手机微信扫码, then say 电脑浏览器请不要使用IE、edge等自带浏览器，可以使用搜狗、谷歌、360浏览器极速模式等浏览器试试。
Otherwise, say 目前支持的学习方式是电脑浏览器或者手机微信扫码两种，建议您再使用正确的方式试试
If the user's used the right method but still has problems, then say 建议清除浏览器或者微信缓存再试试
If the user used the right method and 清除了缓存, then say，抱歉，您的问题涉及到测试，建议您联系平台的人工热线客服或者在线客服进行反馈

Answer the question in Chinese.

{chat_history}
Question: {input}
"""

MULTIPLE_LOGIN_PROMPT = """Answer the user's question step by step. Don't give the whole answer at once. Guide the user to the solution.

Always start with Step 1 below, DO NOT go to Step 2. Only execute Step 1 first. Do Not include the keyword `Step 1` or `Step 2` in your response.

Step 1
First check the user's learning method belongs to 电脑浏览器 or 手机微信扫码

Step 2
Based on the user's choice in Step 1,
If the user's learning method belongs to 电脑浏览器 or 手机微信扫码, then say 请勿使用电脑和手机同时登录账号学习，也不要使用电脑或手机同时登录多人账号学习。
If the user say 没有登录多个账号/没有同时登录 etc., say 建议您清除电脑浏览器或手机微信缓存，并修改平台登录密码后重新登录学习试试。

Answer the question in Chinese.

{chat_history}
Question: {input}
"""

REFUND_PROMPT = """Use a tool to answer the user's qustion.

Ask the user to provide 身份证号，in order to 查询课程信息
You MUST use your ONLY tool to answer the user question.

When user input a number longer than 6 digits, use it as user 身份证号 in the context for the tool.
DO NOT hallucinate!!!! DO NOT Assume any user inputs. ALWAYS ask the user for more information if needed.

You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do.
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Answer the question in Chinese.

Begin!

{chat_history}
Question: {input}
Thought:{agent_scratchpad}
"""

CANT_FIND_COURSE_PROMPT = """Use a tool to answer the user's qustion.

Ask the user to provide 身份证号，in order to 检查用户购买课程记录
You MUST use a tool and generate a response based on tool's output.

When user input a number longer than 6 digits, use it as user 身份证号 in the context for the tool.
DO NOT hallucinate!!!! 

You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do.
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Answer the question in Chinese.

{chat_history}
Question: {input}
Thought:{agent_scratchpad}
"""

MAIN_QUESTION_CLASSIFIER_PROMPT = """根据用户的输入 input 以及对话历史记录 chat_history，判定用户问的内容属于以下哪一类： `学时没显示` 或者 `学时有问题` 或者 `济宁市：如何报班、报名` 或者 `济宁市：课程进度不对` 或者 `济宁市：多个设备，其他地方登录` 或者 `济宁市：课程退款退费，课程买错了` 或者 `济宁市：课程找不到，课程没有了` 或者 `无法登录` 或者 `忘记密码` 或者 `找回密码` 或者 `济宁市` 或者 `注册` 或者 `审核` 或者 `学时对接` 或者 `学时申报` 或者 `学时审核` 或者 `系统操作` 或者 `修改信息` 或者 `其他`.

# 不要回答用户的问题。仅把用户的问题归类为 `学时没显示` 或 `学时有问题` 或 `济宁市：课程进度不对` 或 `济宁市：多个设备，其他地方登录` 或 `济宁市：课程退款退费，课程买错了` 或 `济宁市：课程找不到，课程没有了` 或 `无法登录` 或 `忘记密码` 或 `找回密码` 或 `济宁市` 或 `注册` 或 `审核` 或 `学时对接` 或 `学时申报` 或 `学时审核` 或 `系统操作` 或 `修改信息` 或 `其他`.
# 不要回答除此之外的任何内容。

以下是一些例子：
- "学时没显示" -> 分类为 `学时没显示`
- "学时有问题" -> 分类为 `学时有问题`
- "学时没对接" -> 分类为 `学时没显示`
- "学时没对接" -> 分类为 `学时没显示`
- "学时不对接" -> 分类为 `学时没显示`
- "学时不对" -> 分类为 `学时有问题`
- "学时对接" -> 分类为 `学时对接`
- "学时报错了" -> 分类为 `学时申报`
- "账号密码是什么" -> 分类为 `注册`
- "学时申报" -> 分类为 `学时申报`
- "济宁市，如何补学" -> 分类为 `济宁市`

注意：
如果用户提到了 "济宁市"，你应该将其分类为与 `济宁市` 相关。如果用户没有提到 "济宁市"，你绝对不能将其分类为与 `济宁市` 相关。
如果用户提到了以下具体的济宁市问题，你应该将其分类到济宁市具体的问题中。其他具体问题，统一归类为 `济宁市`。
- "济宁市：课程进度不对" -> 分类为 `济宁市：课程进度不对`
- "济宁市：多个设备，其他地方登录" -> 分类为 `济宁市：多个设备，其他地方登录`
- "济宁市：课程退款退费，课程买错了" -> 分类为 `济宁市：课程退款退费，课程买错了`
- "济宁市：课程找不到，课程没有了" -> 分类为 `济宁市：课程找不到，课程没有了`

{chat_history}
Question: {input}

Classification:"""
