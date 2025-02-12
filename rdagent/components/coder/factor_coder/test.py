from jinja2 import Template
from time import time

# Step 1: 读取模板内容
with open('/home/tangziyi/RD-Agent/rdagent/components/coder/factor_coder/template_debug.jinjia2', 'r') as f:
    template_content = f.read()

# Step 2: 渲染模板
template = Template(template_content)
rendered_code = template.render(
    expression="(POW((INV((((LOG(INV(((($high+$low+$close+$open)/4)+2.0)))-2.0)/10.0)(-0.5)))/(-0.5))+(-0.5)),10.0)", # "DELAY($high + $low / 2, 5)",
    factor_name="FACTOR_1"
    )

# Step 3: 打印渲染后的代码
print(rendered_code)
t0 = time()
exec(rendered_code)
print("TIME COST: ", round(time() - t0, 3))