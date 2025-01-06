from jinja2 import Template

# Step 1: 读取模板内容
with open('/home/tangziyi/RD-Agent/rdagent/components/coder/factor_coder/template_debug.jinjia2', 'r') as f:
    template_content = f.read()

# Step 2: 渲染模板
template = Template(template_content)
rendered_code = template.render(
    expression="RANK(DELTA($volume, 1)) / (STD($close, 30) + 1e-8)", # "DELAY($high + $low / 2, 5)",
    factor_name="FACTOR_1"
    )

# Step 3: 打印渲染后的代码
print(rendered_code)
exec(rendered_code)