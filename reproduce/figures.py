"""Regenerate manuscript figures and tables from frozen results, without model calls."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results/figure_data"
ORDER = ["Cu_alloys__EC", "HEA_Invar__TEC", "MPEA__UTS", "mg_alloy__UTS_newseed",
         "Ti_alloy__YS", "Ti_alloy__UTS", "mg_alloy__YS", "2xxx_T6__EL", "2xxx_T6__UTS"]
LABELS = ["Cu EC", "HEA Invar TEC", "MPEA UTS", "Mg UTS", "Ti YS", "Ti UTS", "Mg YS", "Al EL", "Al UTS"]
LABEL = dict(zip(ORDER, LABELS))
UNITS = ["EC (% IACS)", r"TEC ($10^{-6}$ K$^{-1}$)", "UTS (MPa)", "UTS (MPa)",
         "YS (MPa)", "UTS (MPa)", "YS (MPa)", "EL (%)", "UTS (MPa)"]
STRATEGIES = ["Random", "GPR-UCB", "RFR-UCB", "GBDT-UCB", "NN-UCB", "Adaptive-UCB",
              "LLM-rank", "LLM-dynamic", "Hybrid-static", "Hybrid"]
DISPLAY = {"Hybrid": "CALM", "Hybrid-static": "CALM-static"}
COLORS = dict(zip(STRATEGIES, ["#687482", "#8C564B", "#4DBBD5", "#2F7DCC", "#00A087",
                              "#3C5488", "#D65BA6", "#7E57C2", "#F39B35", "#E64B35"]))
MARKERS = dict(zip(STRATEGIES, ["s", "d", "^", "D", "o", "s", "v", "v", "P", "*"]))
VARIANTS = ["full", "static", "fixed_03", "static_fixed_03", "spearman"]
VARIANT_LABELS = ["Full CALM", "FSR disabled", "EWRC disabled", "FSR and EWRC disabled", "Spearman weighting"]
VNAME = dict(zip(VARIANTS, VARIANT_LABELS))
VCOLOR = dict(zip(VARIANTS, ["#E64B35", "#F39B35", "#3C5488", "#7E8790", "#7E57C2"]))
FIGURES = [str(i) for i in range(1, 10)] + ["S18", "S19", "S20", "S21", "S22"]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_csv(path, rows):
    rows = list(rows)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def save(fig, out, stem):
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(out / f"{stem}.{suffix}", dpi=300, facecolor="white")
    plt.close(fig)


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#E5EBEF", lw=.55)
    ax.set_axisbelow(True)


def panel(ax, letter, title=""):
    ax.set_title(title, loc="left", fontsize=9, fontweight="bold", pad=10)
    ax.text(-.10, 1.06, letter, transform=ax.transAxes, fontsize=11, fontweight="bold")


def load_benchmark():
    summary = read(ROOT / "results/benchmark/summary.json")
    by_id = {t["id"]: t for t in summary["tasks"]}
    return [(by_id[task], read(ROOT / by_id[task]["source_file"])) for task in ORDER]


def means_matrix(benchmark, supplementary=False):
    strategies = STRATEGIES if supplementary else [s for s in STRATEGIES if s != "Hybrid-static"]
    means = np.array([[task["strategies"][s]["normalized_regret_auc_mean"]
                       for task, _ in benchmark] for s in strategies])
    order = np.argsort(means.mean(axis=1), kind="stable")
    strategies = [strategies[i] for i in order]
    means = means[order]
    ranks = pd.DataFrame(means).rank(axis=0, method="average").to_numpy()
    return strategies, means, ranks


def figure_curves(benchmark, out, supplementary=False):
    strategies = STRATEGIES if supplementary else [s for s in STRATEGIES if s != "Hybrid-static"]
    fig, axes = plt.subplots(3, 3, figsize=(9.1, 7.7))
    fig.subplots_adjust(left=.09, right=.985, bottom=.12, top=.95, hspace=.44, wspace=.34)
    values = []
    for j, ((task, result), ax) in enumerate(zip(benchmark, axes.flat)):
        config = result["config"]
        key = "best_" + config["target_col"].lower()
        n = config["n_samples"]
        sizes = np.array([r["train_size"] for r in result["strategies"]["Hybrid"][0]])
        x = sizes * 100 / n
        for name in strategies:
            y = np.array([[r[key] for r in run] for run in result["strategies"][name]])
            mean, sd = y.mean(axis=0), y.std(axis=0, ddof=1)
            ax.plot(x, mean, color=COLORS[name], marker=MARKERS[name], markersize=3.2,
                    markevery=2, lw=1.5 if name == "Hybrid" else .9,
                    ls="--" if name in {"Random", "GPR-UCB", "Adaptive-UCB", "Hybrid-static"} else "-",
                    label=DISPLAY.get(name, name), zorder=5 if name == "Hybrid" else 3)
            ax.fill_between(x, mean - sd, mean + sd, color=COLORS[name], alpha=.10, lw=0)
            for k, fraction in enumerate(x):
                values.append(dict(task=task["id"], strategy=DISPLAY.get(name, name),
                                   measured_fraction_pct=float(fraction), mean=float(mean[k]), sample_sd=float(sd[k])))
        minimize = config["objective_direction"] == "minimize"
        optimum = config["global_min"] if minimize else config["global_max"]
        ax.axhline(optimum, color="#333333", ls=":", lw=.8)
        panel(ax, chr(97+j), LABELS[j] + (" ↓" if minimize else " ↑"))
        ax.set(xlabel="Measured pool (%)", ylabel=UNITS[j], xlim=(9, 41))
        style(ax)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False, fontsize=8,
               bbox_to_anchor=(.52, .01), handlelength=2)
    stem = "figure_S19" if supplementary else "figure_03"
    write_csv(out / f"{stem}_values.csv", values)
    save(fig, out, stem)


def figure_comparison(benchmark, out, supplementary=False):
    strategies, means, ranks = means_matrix(benchmark, supplementary)
    average = means.mean(axis=1)
    efficiency = (100 * means.min(axis=0)[None, :] / means).min(axis=1)
    fig, (ax, heat) = plt.subplots(1, 2, figsize=(9.4, 4.5), gridspec_kw={"width_ratios": [1, 1.4]})
    fig.subplots_adjust(left=.08, right=.985, bottom=.24, top=.88, wspace=.42)
    panel(ax, "a", "Search efficiency and worst-task performance")
    for i, name in enumerate(strategies):
        ax.scatter(average[i], efficiency[i], color=COLORS[name], marker=MARKERS[name], s=65 if name == "Hybrid" else 35, zorder=4)
        offset={"GBDT-UCB":(5,12),"RFR-UCB":(5,-3),"GPR-UCB":(5,4),
                "Hybrid-static":(-5,-12)}.get(name,(5,5))
        ax.annotate(DISPLAY.get(name, name), (average[i], efficiency[i]), xytext=offset,
                    textcoords="offset points", fontsize=7, ha="right" if name=="Hybrid-static" else "left")
    baseline = [i for i,s in enumerate(strategies) if s != "Hybrid"]
    ax.axvline(min(average[baseline]), color="#999999", ls="--", lw=.8)
    ax.axhline(max(efficiency[baseline]), color="#999999", ls="--", lw=.8)
    ax.set(xlabel="Task-averaged nAUC (lower is better)", ylabel="Worst-task relative efficiency (%)", ylim=(0, 103))
    ax.margins(x=.22)
    style(ax)
    panel(heat, "b", "Within-task ranks")
    heat.imshow(ranks, cmap="Blues_r", vmin=1, vmax=len(strategies)+2, aspect="auto")
    heat.set(xticks=range(9), xticklabels=LABELS, yticks=range(len(strategies)),
             yticklabels=[DISPLAY.get(s,s) for s in strategies])
    heat.tick_params(axis="x", rotation=45, labelsize=7)
    for tick in heat.get_xticklabels(): tick.set_ha("right")
    for i in range(len(strategies)):
        for j in range(9):
            heat.text(j,i,f"{ranks[i,j]:g}",ha="center",va="center",fontsize=8,
                      color="white" if ranks[i,j] <= 3 else "#172A3A")
    calm = strategies.index("Hybrid")
    heat.add_patch(Rectangle((-.5,calm-.5),9,1,fill=False,color=COLORS["Hybrid"],lw=1.8))
    for j in (4,5):heat.add_patch(Rectangle((j-.5,-.5),1,len(strategies),fill=False,color="#536778",lw=1,ls="--"))
    stem = "figure_S20" if supplementary else "figure_04"
    write_csv(out / f"{stem}_values.csv", [dict(strategy=DISPLAY.get(s,s),task=ORDER[j],
              mean_nauc=float(means[i,j]),rank=float(ranks[i,j]),task_averaged_nauc=float(average[i]),
              worst_task_relative_efficiency_pct=float(efficiency[i]))
              for i,s in enumerate(strategies) for j in range(9)])
    save(fig,out,stem)


def ablation_values(data):
    rows = data["per_run"]
    return {v:float(np.mean([np.mean([r["nauc"] for r in rows if r["variant"]==v and r["task"]==task["id"]])
                              for task in data["tasks"]])) for v in VARIANTS}


def discovery_curves(data):
    sizes = {t["id"]:t["n_samples"] for t in data["tasks"]}
    curves = {}
    for variant in VARIANTS:
        runs = [r for r in data["per_run"] if r["variant"]==variant]
        initial = sum(r["initial_optimum_hit"] for r in runs)
        events = sorted(100*(r["first_hit_samples"]-1)/sizes[r["task"]]+1e-8
                        for r in runs if r["hit_at_budget"] and not r["initial_optimum_hit"])
        curves[variant] = ([10,*events,40], [initial/len(runs)*100,
                           *[(initial+i+1)/len(runs)*100 for i in range(len(events))],
                           (initial+len(events))/len(runs)*100])
    return curves


def figure_ablation(data,out):
    values=ablation_values(data)
    fig,(cards,ax)=plt.subplots(2,1,figsize=(7.3,8.2),gridspec_kw={"height_ratios":[1,1.2]})
    fig.subplots_adjust(left=.13,right=.96,bottom=.1,top=.95,hspace=.28)
    cards.set(xlim=(0,1),ylim=(0,1));cards.axis("off")
    panel(cards,"a","Component contributions")
    positions={"static_fixed_03":(.23,.76),"fixed_03":(.77,.76),"static":(.23,.2),"full":(.77,.2)}
    for variant,(x,y) in positions.items():
        cards.add_patch(FancyBboxPatch((x-.2,y-.16),.4,.32,boxstyle="round,pad=.01",lw=.7,
                         edgecolor="#D7DCE1",facecolor=VCOLOR[variant]+"0A"))
        title=VNAME[variant].replace("FSR and EWRC disabled","FSR and EWRC\ndisabled")
        cards.text(x,y+.075,title,ha="center",va="center",fontsize=9)
        fsr=variant in ("full","fixed_03");ewrc=variant in ("full","static")
        cards.text(x,y-.015,f"FSR {'✓' if fsr else '×'}     EWRC {'✓' if ewrc else '×'}",ha="center",fontsize=9,color="#62676C")
        cards.text(x,y-.105,f"nAUC  {values[variant]:.5f}",ha="center",fontsize=11)
        cards.scatter([x-.16],[y+.08],color=VCOLOR[variant],s=18)
    effects=[]
    for a,b,module in [("static_fixed_03","fixed_03","FSR"),("static","full","FSR"),
                       ("static_fixed_03","static","EWRC"),("fixed_03","full","EWRC")]:
        reduction=100*(values[b]-values[a])/values[a]
        x1,y1=positions[a];x2,y2=positions[b]
        if y1==y2:
            cards.annotate("",(x2-.22,y2),(x1+.22,y1),arrowprops={"arrowstyle":"->","color":"#777777"})
            cards.text(.5,y1+.08,f"Enable {module}",ha="center",fontsize=8)
            cards.text(.5,y1-.08,f"{reduction:.1f}%",ha="center",fontsize=9)
        else:
            cards.annotate("",(x2,y2+.18),(x1,y1-.18),arrowprops={"arrowstyle":"->","color":"#777777"})
            cards.text(x1+(.02 if x1<.5 else -.02),.48,f"Enable {module}\n{reduction:.1f}%",
                       ha="left" if x1<.5 else "right",va="center",fontsize=8)
        effects.append(dict(start=VNAME[a],end=VNAME[b],module=module,relative_nauc_change_pct=reduction))
    panel(ax,"b","Optimum discovery")
    for variant,(x,y) in discovery_curves(data).items():
        if variant=="spearman":continue
        ax.step(x,y,where="post",lw=1.8 if variant=="full" else 1.3,color=VCOLOR[variant],
                ls="--" if variant in ("static","static_fixed_03") else "-",label=VNAME[variant])
    ax.set(xlim=(10,40),ylim=(0,102),xlabel="Nominal measurement budget (% of pool)",ylabel="Runs finding the optimum (%)")
    ax.legend(frameon=False,loc="lower right",fontsize=8);style(ax)
    write_csv(out/"figure_05_effects.csv",effects)
    save(fig,out,"figure_05")


def figure_knowledge(out):
    data=pd.read_csv(DATA/"knowledge_ranks.csv")
    static=[data.loc[data.task==t,"static_spearman"].to_numpy() for t in ORDER]
    updated=[data.loc[data.task==t,"updated_spearman"].to_numpy() for t in ORDER]
    gains=[data.loc[data.task==t,"pair_accuracy_gain_pp"].to_numpy() for t in ORDER]
    fig,(a,b)=plt.subplots(2,1,figsize=(7.2,5.7))
    fig.subplots_adjust(left=.12,right=.98,bottom=.13,top=.94,hspace=.42)
    x=np.arange(9)
    for vals,offset,color,label in [(static,-.19,"#B7A1C9","Static"),(updated,.19,"#D96D6B","Updated")]:
        a.bar(x+offset,[v.mean() for v in vals],.34,color=color,label=label,
              yerr=[v.std(ddof=1) for v in vals],error_kw={"elinewidth":.8,"capsize":2})
    a.set(xticks=x,xticklabels=LABELS,ylabel="Shortlist Spearman correlation")
    a.legend(frameon=False,ncol=2,loc="lower center",bbox_to_anchor=(.5,1.02));panel(a,"a");style(a)
    boxes=b.boxplot(gains,positions=x,widths=.45,whis=(0,100),patch_artist=True,showfliers=False)
    for box in boxes["boxes"]:box.set(facecolor="#F7E3E2",edgecolor="#D96D6B")
    for line in boxes["medians"]:line.set(color="#A33F4C",lw=1.3)
    for i,vals in enumerate(gains):
        b.scatter(i+np.linspace(-.1,.1,len(vals)),vals,color="#D96D6B",s=14,zorder=4)
        b.scatter([i],[vals.mean()],marker="D",color="#8C6DA5",s=24,zorder=5)
    b.axhline(0,color="#777777",lw=.7,ls="--")
    b.set(xticks=x,xticklabels=LABELS,ylabel="Pairwise accuracy change (pp)")
    panel(b,"b");style(b)
    for ax in (a,b):
        ax.tick_params(axis="x",labelsize=7)
        ax.set_xticklabels([s.replace(" ","\n",1) for s in LABELS])
    save(fig,out,"figure_06")


def figure_case(out):
    data=read(DATA/"mpea_case.json")
    fig=plt.figure(figsize=(7.6,6.9))
    outer=fig.add_gridspec(2,1,height_ratios=[1.45,1],hspace=.43,left=.09,right=.97,top=.94,bottom=.12)
    top=outer[0].subgridspec(1,3,width_ratios=[1.1,1.25,.7],wspace=.28)
    comp=fig.add_subplot(top[0]);quotes=fig.add_subplot(top[1]);rank=fig.add_subplot(top[2])
    elements=["Mo","Nb","Ti","V","Zr"]
    for key,color,label in [("candidate","#4C86C6",r"MoNbTiV$_{0.75}$Zr"),("local_reference_M1","#F08B51",r"MoNbTiV$_2$Zr (M$_1$)")]:
        vals=[data[key]["composition_at_percent"][e] for e in elements]
        comp.plot(elements,vals,"o-",color=color,lw=1.2,ms=4,label=label)
    comp.set(ylabel="Composition (at.%)",ylim=(0,40));style(comp)
    comp.legend(frameon=False,fontsize=7,loc="upper center",bbox_to_anchor=(.5,-.09))
    panel(comp,"a","Candidate and local reference")
    quotes.axis("off")
    for y,title,quote,color in [(.9,"Initial LLM","… plausible but not\nnear-optimal.","#F1F1F1"),
                                (.55,"M1 selection","… same refractory BCC\nalloy family …","#EAF6EF"),
                                (.18,"Updated LLM","… slightly below the best\nanchors but above M1.","#FFF0E6")]:
        quotes.text(.04,y,title,fontweight="bold",fontsize=8,transform=quotes.transAxes)
        quotes.text(.04,y-.14,quote,fontsize=8,va="center",transform=quotes.transAxes,
                    bbox={"facecolor":color,"edgecolor":"none","boxstyle":"round,pad=.5"})
    ranks=data["scores_and_ranks"]
    vals=[ranks["same_state_static_rank"],ranks["same_state_dynamic_rank"]]
    rank.plot([0,1],vals,"o-",color="#AE1BBC",lw=1.7,ms=6)
    rank.set(xticks=[0,1],xticklabels=["Static","Updated"],ylim=(310,-10),xlim=(-.3,1.3),
             yticks=[1,50,100,150,200,250,297],ylabel="Knowledge rank")
    for x,v in enumerate(vals):rank.annotate(str(v),(x,v),xytext=(0,10),textcoords="offset points",ha="center",fontweight="bold")
    rank.text(.5,-.17,f"Same {ranks['rank_denominator']} candidates",ha="center",transform=rank.transAxes,fontsize=8)
    style(rank)
    lower=outer[1].subgridspec(1,2,wspace=.42)
    measured=fig.add_subplot(lower[0]);acq=fig.add_subplot(lower[1])
    strength=[data["local_reference_M1"]["measured_value"],data["shared_feedback_batch5"]["best_anchor_value_display"],data["candidate"]["measured_value"]]
    for i,(v,color) in enumerate(zip(strength,["#4C86C6","#7394BA","#008F83"])):
        measured.vlines(i,0,v,color=color,lw=1.5);measured.scatter([i],[v],color=color,s=40)
        measured.text(i,v+120,f"{v:g}",ha="center",fontsize=9)
    measured.set(xticks=range(3),xticklabels=["Local reference\n(M1)","Previous\nbest","Candidate\n(C131)"],ylabel="Strength (MPa)",ylim=(0,4500),xlim=(-.5,2.5))
    panel(measured,"b","Measured strength");style(measured)
    old,new=data["acquisition"]["static_rank_lower_bound"],data["acquisition"]["updated_rank"]
    acq.axhspan(1,7,color="#DBF2E8");acq.axhline(7,color="#008F83",ls="--",lw=.8)
    acq.plot([0,1],[old,new],"o-",color="#AE1BBC",lw=1.7,ms=6)
    for x,v,label in [(0,old,f"≥{old}"),(1,new,str(new))]:acq.annotate(label,(x,v),xytext=(10,0),textcoords="offset points",fontsize=10)
    acq.text(.5,3.5,"7 query slots",ha="center",color="#008F83")
    acq.set(xticks=[0,1],xticklabels=["Static\nknowledge","Updated\nknowledge"],ylim=(22,0),
            yticks=[1,7,13,19],xlim=(-.3,1.4),ylabel="Acquisition rank")
    panel(acq,"c","Same model and knowledge weight");style(acq)
    save(fig,out,"figure_07")


def search_panel(ax,data,name,letter):
    pool={p["candidate_index"]:p for p in data["pool"]}
    ax.scatter([p["PC1"] for p in pool.values()],[p["PC2"] for p in pool.values()],s=8,color="#D9DFE5",zorder=1)
    rounds=data["case"][name]["rounds"];hit=data["case"][name]["first_hit_batch"]
    palette=["#3264BE","#009C9E","#DA972D","#D65364","#8A56A6","#84643D","#C84C91","#19755C","#749DCF","#87972B","#E17B3D","#404B64","#7E8E9C","#A396AB","#789C9B","#B49C7E"]
    for row in rounds[1:]:
        batch=row["batch"];points=[pool[i] for i in row["indices"]]
        ax.scatter([p["PC1"] for p in points],[p["PC2"] for p in points],s=16,color=palette[(batch-1)%len(palette)],alpha=1 if batch<=hit else .2,zorder=3)
    centers=np.array([r["centroid"] for r in rounds[:hit+1]])
    for p,q in zip(centers,centers[1:]):
        arrow=ax.annotate("",q,p,annotation_clip=False,
                          arrowprops={"arrowstyle":"->","color":"#444444","lw":.8})
        arrow.arrow_patch.set_clip_path(ax.patch)
    ax.scatter(*centers[0],marker="s",s=40,color="white",edgecolor="#222222",zorder=7)
    offsets={"Hybrid-static":{1:(13,-10),2:(-13,13),3:(4,18),4:(-12,20),6:(-6,-20),
              7:(14,-9),8:(22,-15),9:(13,-30),10:(-9,-11),11:(-11,-22),12:(7,-31)},
             "Hybrid":{1:(11,2),2:(-12,-8),3:(5,14),4:(5,-16),5:(-10,11)}}
    for batch,(x,y) in enumerate(centers[1:],1):
        if not (-4.25<=x<=4.35 and -1.06<=y<=.81):continue
        color=palette[(batch-1)%len(palette)]
        ax.scatter([x],[y],s=9,color=color,zorder=6)
        ax.annotate(str(batch),(x,y),xytext=offsets[name].get(batch,(4,4)),textcoords="offset points",
                    fontsize=7,zorder=8,ha="center",va="center",
                    bbox={"boxstyle":"circle,pad=.22","facecolor":"white","edgecolor":color},
                    arrowprops={"arrowstyle":"-","color":color,"lw":.65})
    optimum=pool[data["global_optimum_indices"][0]]
    ax.scatter([optimum["PC1"]],[optimum["PC2"]],marker="*",s=120,color="#E5B647",edgecolor="#222222",lw=.5,zorder=9)
    ax.set(xlim=(-4.25,4.35),ylim=(-1.06,.81),xlabel="PC1",ylabel="PC2")
    panel(ax,letter,"Full CALM" if name=="Hybrid" else "FSR disabled");style(ax)
    inset=ax.inset_axes([.73,.72,.25,.25])
    inset.scatter([p["PC1"] for p in pool.values()],[p["PC2"] for p in pool.values()],s=2,color="#C8CED4")
    inset.plot(centers[:,0],centers[:,1],"o-",ms=2,lw=.6,color=VCOLOR["full"] if name=="Hybrid" else VCOLOR["static"])
    for batch,(x,y) in enumerate(centers[1:],1):
        if y>.81:inset.annotate(str(batch),(x,y),xytext=(3,3),textcoords="offset points",fontsize=6)
    inset.set(xlim=(-4.5,4.6),ylim=(-1.7,29.6),xticks=[],yticks=[])
    inset.set_title("Full pool",fontsize=6,pad=2)


def search_curve(ax,data,names):
    names_map={"Hybrid":"Full CALM","Hybrid-static":"FSR disabled","Adaptive-UCB":"Adaptive-UCB"}
    colors={"Hybrid":"#E64B35","Hybrid-static":"#F39B35","Adaptive-UCB":"#343C45"}
    for name in names:
        case=data["case"][name];rounds=case["rounds"]
        ax.step([r["train_size"] for r in rounds],[r["best"] for r in rounds],where="post",color=colors[name],lw=1.5,label=names_map[name])
        ax.scatter([case["first_hit_labels"]],[data["global_optimum"]],marker="*",s=100,color=colors[name],edgecolor="white",lw=.5,zorder=5)
        ax.annotate(str(case["first_hit_labels"]),(case["first_hit_labels"],data["global_optimum"]),xytext=(0,10),textcoords="offset points",ha="center",fontsize=8,color=colors[name])
    ax.axhline(data["global_optimum"],color="#888888",ls=":",lw=.8)
    ax.set(xlabel="Measured candidates (including initialization)",ylabel="Best strength (MPa)",xlim=(35,147),ylim=(3100,4050))
    ax.legend(frameon=False,fontsize=8,loc="lower right");style(ax)


def figure_search(out,supplementary=False):
    data=read(DATA/"mpea_search.json")
    if supplementary:
        fig,ax=plt.subplots(figsize=(7,3.5));fig.subplots_adjust(left=.12,right=.97,bottom=.2,top=.88)
        search_curve(ax,data,["Hybrid","Adaptive-UCB","Hybrid-static"])
        ax.set_title("MPEA UTS · paired run 7",loc="left",fontsize=10)
        save(fig,out,"figure_S21");return
    fig=plt.figure(figsize=(7.6,7.1))
    gs=fig.add_gridspec(2,2,height_ratios=[1.2,1],left=.10,right=.97,bottom=.10,top=.94,wspace=.3,hspace=.4)
    search_panel(fig.add_subplot(gs[0,0]),data,"Hybrid-static","a")
    search_panel(fig.add_subplot(gs[0,1]),data,"Hybrid","b")
    ax=fig.add_subplot(gs[1,:]);search_curve(ax,data,["Hybrid","Hybrid-static"])
    ax.axvspan(65,72,color="#E64B35",alpha=.08)
    panel(ax,"c","Best observed strength")
    save(fig,out,"figure_08")


def figure_ewrc(out):
    data=read(DATA/"ewrc_selection.json")
    queries=pd.read_csv(DATA/"ewrc_queries.csv")
    queries["changed"]=[set(json.loads(a))!=set(json.loads(b)) for a,b in zip(queries.actual_selected,queries.old_weight_selected)]
    order=["Cu_alloys__EC","HEA_Invar__TEC","MPEA__UTS","Ti_alloy__YS","Ti_alloy__UTS","mg_alloy__UTS_newseed","mg_alloy__YS","2xxx_T6__EL","2xxx_T6__UTS"]
    counts=[dict(task=t,n=int((queries.task==t).sum()),changed=int(queries.loc[queries.task==t,"changed"].sum())) for t in order]
    counts.append(dict(task="All states",n=len(queries),changed=int(queries.changed.sum())))
    write_csv(out/"figure_09_counts.csv",counts)
    fig=plt.figure(figsize=(7.5,8.2));outer=fig.add_gridspec(2,2,left=.12,right=.97,bottom=.08,top=.94,wspace=.43,hspace=.38)
    ax=fig.add_subplot(outer[0,0]);panel(ax,"a","Selection changes across tasks")
    y=np.arange(10);pct=[100*r["changed"]/r["n"] for r in counts]
    ax.hlines(y,0,pct,color="#F2B4A9",lw=2);ax.scatter(pct,y,color="#E64B35",s=24)
    for i,r in enumerate(counts):ax.text(105,i,f"{r['changed']}/{r['n']}",fontsize=7,va="center")
    ax.set(yticks=y,yticklabels=[LABEL.get(r["task"],r["task"]) for r in counts],xlim=(0,132),ylim=(9.7,-.7),
           xticks=[0,50,100],xlabel="Queries with changed selection (%)")
    ax.tick_params(axis="y",labelsize=7)
    ax.text(.5,1.01,"218 eligible states · 55 runs · 36.7% changed",transform=ax.transAxes,ha="center",fontsize=7)
    style(ax)
    for i,(trajectory,cell) in enumerate(zip(data["full_run_trajectories"],[outer[0,1],outer[1,0],outer[1,1]])):
        inner=cell.subgridspec(2,1,height_ratios=[2,1],hspace=.15)
        weight=fig.add_subplot(inner[0]);best=fig.add_subplot(inner[1],sharex=weight)
        b=trajectory["example_batch"];weights=trajectory["weights"]
        weight.plot(trajectory["weight_batches"],weights,"o-",color="#216C78",ms=3,lw=1)
        weight.plot([b-1,b],weights[b-2:b],"o-",color="#E64B35",ms=4,lw=1.8)
        weight.set(ylim=(0,.6),ylabel="Pre-query weight λ")
        title=["Weight increase at B2","Weight decrease at B8","Weight restored at B2"][i]
        panel(weight,"bcd"[i],title)
        weight.text(.5,1.01,f"{LABEL[trajectory['task']]} · Run {trajectory['run']}",transform=weight.transAxes,ha="center",fontsize=7)
        best.step(trajectory["best_batches"],trajectory["best"],where="post",color="#171C20",lw=1.1)
        best.plot(trajectory["best_batches"],trajectory["best"],"o",color="#171C20",ms=2)
        best.set(xlabel="Query batch",ylabel="Best (MPa)",ylim=trajectory["best_axis_limits"],xlim=(0,15.5),xticks=sorted({0,5,10,15,b}))
        plt.setp(weight.get_xticklabels(),visible=False)
        for axis in (weight,best):axis.axvspan(b-.35,b+.35,color="#FCF0E7",zorder=0);style(axis)
    save(fig,out,"figure_09")


def figure_first_batch(out):
    data=read(DATA/"first_batch.json")
    names=["CALM","Adaptive-UCB","NN-UCB","LLM-dynamic"]
    fig,axes=plt.subplots(1,2,figsize=(8,4),sharey=True)
    fig.subplots_adjust(left=.1,right=.98,bottom=.25,top=.82,wspace=.12)
    for i,(scope,ax) in enumerate(zip(["all_runs","exclude_initial_global_optimum"],axes)):
        for j,name in enumerate(names):
            vals=[data["regions"][q]["scopes"][scope]["aggregate"][name]["precision_first_1_to_5_batches_pct"][0] for q in ("20","10","5")]
            ax.bar(np.arange(3)+(j-1.5)*.2,vals,.18,color=COLORS["Hybrid" if name=="CALM" else name],label=name)
        ax.set(xticks=range(3),xticklabels=["Top 20%","Top 10%","Top 5%"],ylim=(0,100))
        panel(ax,"ab"[i],"All 90 paired runs" if i==0 else "Excluding initial optima (83 runs)");style(ax)
    axes[0].set_ylabel("First-batch acquisition fraction (%)")
    fig.legend(*axes[0].get_legend_handles_labels(),ncol=4,loc="lower center",frameon=False,fontsize=8)
    save(fig,out,"figure_S18")


def figure_reliability(out):
    data=read(DATA/"knowledge_reliability.json")
    rows={r["task"]:r for r in data["per_task"]}
    fig,axes=plt.subplots(1,2,figsize=(7.5,5.7),sharey=True)
    fig.subplots_adjust(left=.2,right=.97,bottom=.15,top=.86,wspace=.22)
    for ax,target,letter in zip(axes,["EWRC","Spearman"],"ab"):
        a=[rows[t][f"correlation_abs_error_target_{target}_estimate_EWRC"] for t in ORDER]
        b=[rows[t][f"correlation_abs_error_target_{target}_estimate_Spearman"] for t in ORDER]
        for y,(x,z) in enumerate(zip(a,b)):ax.plot([x,z],[y,y],color="#C9CFD6",lw=1)
        ax.scatter(a,range(9),color="#E64B35",s=28,label="EWRC")
        ax.scatter(b,range(9),edgecolor="#7E57C2",facecolor="white",marker="s",s=28,label="Spearman")
        ax.barh(10,np.mean(a),height=.32,color="#E64B35")
        ax.barh(10.4,np.mean(b),height=.32,color="#7E57C2")
        ax.set(yticks=list(range(9))+[10.2],yticklabels=LABELS+["Task mean"],ylim=(11,-.7),xlim=(0,.42),xlabel="Mean absolute correlation error")
        panel(ax,letter,("Extreme-weighted" if target=="EWRC" else "Ordinary Spearman")+" reference")
        style(ax)
    fig.legend(*axes[0].get_legend_handles_labels(),ncol=2,loc="upper center",frameon=False,bbox_to_anchor=(.59,.98))
    save(fig,out,"figure_S22")


def tables(benchmark,ablation,out):
    means=ablation_values(ablation);sizes={t["id"]:t["n_samples"] for t in ablation["tasks"]}
    table=[]
    for variant in VARIANTS:
        runs=[r for r in ablation["per_run"] if r["variant"]==variant]
        row={"configuration":VNAME[variant],"mean_nauc":means[variant]}
        for pct in (10,20,30,40):
            row[f"hit_at_{pct}pct_percent"]=100*np.mean([r["hit_at_budget"] and r["first_hit_samples"]<=math.ceil(sizes[r["task"]]*pct/100) for r in runs])
        table.append(row)
    write_csv(out/"Table_01.csv",table)
    write_csv(out/"Table_02.csv",[dict(task=LABEL[t["id"]],direction=c["config"]["objective_direction"],
              candidates=c["config"]["n_samples"],initial=c["config"]["n_init"],batch=c["config"]["batch_size"],
              total_budget=c["config"]["total_budget"],paired_runs=len(c["strategies"]["Hybrid"])) for t,c in benchmark])
    first=read(DATA/"first_batch.json")
    for batches,stem in [(1,"Table_S15_1"),(5,"Table_S15_2")]:
        rows=[]
        for name in [DISPLAY.get(s,s) for s in STRATEGIES]:
            row={"strategy":name}
            for q in ("20","10","5"):
                for scope,short in [("all_runs","all90"),("exclude_initial_global_optimum","without_initial_optima83")]:
                    row[f"top{q}_{short}_percent"]=first["regions"][q]["scopes"][scope]["aggregate"][name]["precision_first_1_to_5_batches_pct"][batches-1]
            rows.append(row)
        write_csv(out/f"{stem}.csv",rows)
    write_csv(out/"Table_S15_3.csv",[dict(nominal_top_percent=int(q),**r) for q in ("20","10","5") for r in first["regions"][q]["thresholds"]])
    main,_,mr=means_matrix(benchmark);supp,_,sr=means_matrix(benchmark,True)
    write_csv(out/"Table_S16_1.csv",[dict(task=LABEL[t],CALM_rank_nine_strategies=float(mr[main.index("Hybrid"),i]),
                 CALM_rank_ten_strategies=float(sr[supp.index("Hybrid"),i])) for i,t in enumerate(ORDER)])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",type=Path,default=ROOT/"outputs/paper_figures")
    parser.add_argument("--figures",nargs="+",choices=FIGURES,default=FIGURES,help="Figure numbers; default: all main figures and S18-S22.")
    parser.add_argument("--tables-only",action="store_true")
    parser.add_argument("--skip-tables",action="store_true")
    args=parser.parse_args();out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":8,"axes.titlesize":9,
                         "axes.labelsize":8,"xtick.labelsize":7,"ytick.labelsize":7,
                         "axes.linewidth":.7,"pdf.fonttype":42,"ps.fonttype":42,"svg.fonttype":"none"})
    benchmark=load_benchmark();ablation=read(ROOT/"results/ablations.json")
    selected=[] if args.tables_only else args.figures
    for number in selected:
        if number in ("1","2"):
            source=ROOT/"assets"/("calm-overview.png" if number=="1" else "calm-framework.png")
            shutil.copyfile(source,out/f"figure_{int(number):02}.png")
        elif number in ("3","S19"):figure_curves(benchmark,out,number=="S19")
        elif number in ("4","S20"):figure_comparison(benchmark,out,number=="S20")
        elif number=="5":figure_ablation(ablation,out)
        elif number=="6":figure_knowledge(out)
        elif number=="7":figure_case(out)
        elif number in ("8","S21"):figure_search(out,number=="S21")
        elif number=="9":figure_ewrc(out)
        elif number=="S18":figure_first_batch(out)
        elif number=="S22":figure_reliability(out)
        print(f"Saved Figure {number}")
    if not args.skip_tables:
        tables(benchmark,ablation,out)
        print("Saved Tables 1, 2, S15.1-3 and S16.1")
    print(f"Outputs: {out}")


if __name__=="__main__":
    main()
