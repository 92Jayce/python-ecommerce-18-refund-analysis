"""第18讲 Python退款与售后分析；全程Python，使用模拟教学数据。"""
from pathlib import Path
import sys
if "__file__" in globals():
    COURSE_ROOT = Path(__file__).resolve().parents[1]
else:
    COURSE_ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "course_common.py").exists())
if str(COURSE_ROOT) not in sys.path:
    sys.path.insert(0, str(COURSE_ROOT))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from course_common import (ROOT, DATA, AS_OF, BLUE, TEAL, ORANGE, load_tables,
    build_facts, positive_orders, rfm_table, safe_divide, outdir, save_table, save_figure, show)


# %% 付款订单的退款观察时间
t=load_tables();lines,orders=build_facts(t)
paid=orders.loc[orders.paid_total_cents.gt(0)].copy()
paid['mature_21d']=paid.order_date.add(pd.Timedelta(days=21)).le(AS_OF)
rows=[]
for flag,label in [(True,'at_least_21_days'),(False,'under_21_days')]:
    sub=paid.loc[paid.mature_21d.eq(flag)]
    rows.append({'observation_group':label,'paid_orders':len(sub),
        'orders_with_refund':int(sub.refund_total_cents.gt(0).sum()),
        'observed_refund_order_rate':sub.refund_total_cents.gt(0).mean(),
        'observed_refund_amount_rate':safe_divide(sub.refund_total_cents.sum(),sub.paid_total_cents.sum())})
windows=pd.DataFrame(rows);show(windows);save_table(windows,18,'refund_observation_groups')

# %% 只在完成21天观察的付款订单中比较类目
eligible=set(paid.loc[paid.mature_21d,'order_id'])
mature=lines.loc[lines.order_id.isin(eligible)].copy()
category=mature.groupby('category',as_index=False).agg(
    paid_cents=('paid_cents','sum'),refund_cents=('refund_cents','sum'),
    shipped_units=('shipped_units','sum'),returned_units=('refunded_qty','sum'))
category['refund_amount_rate']=safe_divide(category.refund_cents,category.paid_cents)
category['return_unit_rate']=safe_divide(category.returned_units,category.shipped_units)
assert category.refund_cents.sum()==paid.loc[paid.mature_21d,'refund_total_cents'].sum()
assert category.return_unit_rate.between(0,1).all()
show(category);save_table(category,18,'mature_category_refunds')

# %% 用Python画两种不同分母的售后指标
fig,axes=plt.subplots(1,2,figsize=(9,4))
axes[0].bar(category.category,category.refund_amount_rate*100,color=ORANGE)
axes[0].set(title='Mature orders: refund amount rate',ylabel='Refunded CNY / paid CNY (%)')
axes[1].bar(category.category,category.return_unit_rate*100,color=TEAL)
axes[1].set(title='Mature orders: returned-unit rate',ylabel='Returned units / shipped units (%)')
save_figure(fig,18,'refund_rates')
print('同一订单可能涉及多个类目，所以不能简单相加各类目的订单人数。')
print('最近订单尚未完成退款观察；成熟组和近期组构成不同，比例差不是经营效果。')
