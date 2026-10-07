import json
#import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

df = pd.DataFrame(dict(
    x = [1, 3, 2, 4],
    y = [1, 2, 3, 4]
))
#fig = px.line(df, x="x", y="y", title="Unsorted Input")
#fig.write_html("/tmp/plot.html")

with open("out.jsonl", mode='r') as f:
    lines = f.read().splitlines()

# for key in ["M", "N", "mean"]:
#     del data[key]
#for k,v in data.items():
    #print(f"{k} {len(v)}")

fig = go.Figure()
for line in lines:
    data = json.loads(line)
    fig.add_trace(go.Scatter(x=data['hist_bin_edges'][1:], y=data['hist_counts'], mode="lines", name=data['N']))
fig.update_layout(title=dict(text='Dot Products of Random Unit Vectors in Various Dimensions (1 billion samples each)', x=0.5))
fig.update_yaxes(type='log', title_text='sample count (log)')
fig.write_html("/tmp/plot.html")
