def extract_expr(self, code_str: str) -> str:
    """从代码字符串中提取expr表达式"""
    import re
    # 使用正则表达式匹配expr = "xxx"或expr = 'xxx'的模式
    pattern = r'expr\s*=\s*["\']([^"\']*)["\']'
    match = re.search(pattern, code_str)
    if match:
        return match.group(1)
    return None 