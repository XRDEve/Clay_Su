import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# This code analyses the liquidity index of natural clays extracted from the field via the VC method.
# The code also calculates and approximation of the Undrained Shear Strength, Su, based on the liquidity and plasticity
# indices. Classification of Clay type is performed via Clarke (2018).
# @E.Tripoliti 23/05/2025 13:15 PM

file_path = r"E:\Documents\your_excel.xlsx"
df = pd.read_excel(file_path)
df.rename(columns={
    "Plasticity Index, Ip": "PI",
    "Liquid Limit, wL (%)": "LL",
    "Plastic Limit, wp (%)": "PL",
    "Natural Moisture Content, w (%)": "Wn"
}, inplace=True)

# Recalculate PI if missing or invalid: PI = LL - PL.----------------------------------------------------------------
df["PI"] = df["LL"] - df["PL"]

# Calculate Liquidity Index, LI. It is important to know that if: ---------------------------------------------------
        # LI < 0            ->         Stiff soil
        # 0 <= LI <0.5      ->         Stiff to firm
        # 0.5 <= LI < 1     ->         Soft
        # LI ~ 1            ->         At liquid limit, possibly flowing & slurry like ------------------------------
df["LI"] = (df["Wn"] - df["PL"]) / df["PI"]

# Classify clay type based on Liquid Limit.--------------------------------------------------------------------------
def classify_clay(ll):
    if ll < 35:
        return "CL"
    elif 35 <= ll < 50:
        return "CI"
    elif 50 <= ll <= 70:
        return "CH"
    else:
        return "CV"
df["Clay_Type"] = df["LL"].apply(classify_clay)

# Calculate A-line, T-line, U-line.----------------------------------------------------------------------------------
df["A_line"] = 0.73 * (df["LL"] - 20)
df["T_line"] = 0.73 * (df["LL"] - 11)
df["U_line"] = 0.90 * (df["LL"] - 8)

# Su calculation parameters for each clay type, where a & A, non-fixed constants. ----------------------------------
# | Clay Type         |  a   (kPa / % PI)       |  A (kPa)
# | CL (Low PI)       | 0.3 – 0.4               | 20 – 25
# | CI (Medium PI)    | 0.25 – 0.35             | 18 – 22
# | CH (High PI)      | 0.15 – 0.25             | 12 – 18
# | CV (Very High PI) | 0.1 – 0.2               | 8 – 12-
#-------------------------------------------------------------------------------------------------------------------
def get_clay_parameters(Clay_Type, source=""):
    base_params = {
        "CL": {"a": 0.35, "A": 22.5},
        "CI": {"a": 0.25, "A": 20},
        "CH": {"a": 0.2, "A": 15},
        "CV": {"a": 0.15, "A": 10}
    }
    params = base_params.get(Clay_Type, {"a": 0.2, "A": 15})

    # # Conservative adjustment for shallow soils
    # if depth < 1.5:
    #     params["a"] *= 0.8
    #     params["A"] *= 0.75

    # # Optional: Source-based tuning
    # if "CPT" in str(source).upper():
    #     params["A"] *= 1.1
    # elif "LAB" in str(source).upper():
    #     params["a"] *= 1.2
    return params

# Compute Su.----------------------------------------------------------------------------------------------------
def compute_su(row):
    params = get_clay_parameters(row["Clay_Type"])
    # Calculate Su from LI if valid and positive
    if row["LI"] is not None and row["LI"] > 0:
        su_li = params["A"] / row["LI"]
    else:
        su_li = None
    # Use Su from LI if positive, else calculate Su from PI
    if su_li is not None and su_li > 0:
        su = su_li
    else:
        if row["PI"] is not None and row["PI"] > 0:
            su = params["a"] * row["PI"]
        else:
            su = np.nan
    return su
df["Su_kPa"] = df.apply(compute_su, axis=1)
output_cols = [
    "ID", "Dept (m)", "LL", "PL", "PI", "Wn", "LI", "Clay_Type",
    "A_line", "T_line", "U_line", "Su_kPa"
]
print(df[output_cols])
output_path = r"E:\Documents\Clays_Processed.xlsx"
df.to_excel(output_path, index=False)

# ====================================================== PLOTTING ======================================================
# Define LL range for plotting A-, T-, U- lines----------------------------------------------------------------------
LL_range = np.linspace(10, 90, 500)
A_line_plot = 0.73 * (LL_range - 20)
T_line_plot = 0.73 * (LL_range - 11)
U_line_plot = 0.90 * (LL_range - 8)

# Figure 1: Casagrande ----------------------------------------------------------------------------------------------
plt.figure(figsize=(10, 7))
plt.scatter(df["LL"], df["PI"], color='black', s=30, label='Samples')

# Mark points below A-line as Silt/Organic, as pointed out by the literature.----------------------------------------
below_A = df["PI"] <= 0.73 * (df["LL"] - 20)
plt.scatter(df.loc[below_A, "LL"], df.loc[below_A, "PI"],
            facecolors='red', edgecolors='red', marker='s', s=80,
            label='Silt/Organic')

# Mark points above U-line as possible errros, as also pointed out by the literature.--------------------------------
# above_U = df["PI"] >= 0.90 * (df["LL"] - 8)  # CORRECTED HERE
# plt.scatter(df.loc[above_U, "LL"], df.loc[above_U, "PI"],
#             facecolors='red', edgecolors='red', marker='^', s=80,
#             label='Possibly error')


plt.plot(LL_range, A_line_plot, 'k--', label='A-line')
plt.plot(LL_range, U_line_plot, 'k--', label='U-line')
plt.plot(LL_range, T_line_plot, 'k-', linewidth=0.5, label='T-line')

# Add vertical dashed grey lines at LL = 35, 50, 70 for clay type boundaries-----------------------------------------
for bound in [35, 50, 70]:
    plt.axvline(x=bound, color='dimgrey', linestyle='-.', linewidth=0.5)
clay_label_positions = [(17.5, 39), (42.5, 39), (60, 10), (80, 10)]
clay_labels = ['CL', 'CI', 'CH', 'CV']
for pos, label in zip(clay_label_positions, clay_labels):
    plt.text(pos[0], pos[1], label, fontsize=14, color='dimgrey', fontweight='bold',
             ha='center', va='center', alpha=0.7)
plt.text(75, 41.5, 'A-line', fontsize=12, color='black', fontweight='bold', rotation=32)
plt.text(80, 51.5, 'T-line', fontsize=12, color='black', fontweight='bold', rotation=35)
plt.text(65, 53, 'U-line', fontsize=12, color='black', fontweight='bold', rotation=35)
plt.xlim(10, 90)
plt.ylim(-5, 60)
plt.rcParams['font.size'] = '13'
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.xlabel('Liquid Limit, LL (%)', fontsize=13)
plt.ylabel('Plasticity Index, PI (%)', fontsize=13)
# plt.title('Casagrande Plasticity Chart')
plt.legend(loc='upper left')
# plt.grid(True, linestyle='--', alpha=0.5)
plt.grid(False)

plt.tight_layout()
plt.savefig('E:\Documents\Casagrande.png', dpi=600)
plt.show()


# ======================================================= PLOTTING SU ==================================================
# Herein, the approximation of the Su is plotted versus other physical parameters.---------------------------------
fig, axs = plt.subplots(2, 2, figsize=(14, 12))
# 1. Su vs Depth---------------------------------------------------------------------------------------------------
axs[0, 0].scatter(df["Su_kPa"], df["Dept (m)"], label='Su', marker='o')
axs[0, 0].invert_yaxis()
axs[0, 0].set_xlabel("Su (kPa)",fontsize=13)
axs[0, 0].set_ylabel("Depth (m)",fontsize=13)
axs[0, 0].set_title("Su vs Depth",fontsize=13)
axs[0, 0].legend()
axs[0, 0].tick_params(axis='x', labelsize=13)
axs[0, 0].tick_params(axis='y', labelsize=13)
axs[0, 0].grid(False)

# 2. Su distribution by Clay Type (boxplot).------------------------------------------------------------------------
sns.boxplot(x='Clay_Type', y='Su_kPa', data=df, ax=axs[0, 1])
axs[0, 1].set_title("Su Distribution by Clay Type",fontsize=13)
axs[0, 1].set_xlabel("Clay Type",fontsize=13)
axs[0, 1].set_ylabel("Su (kPa)",fontsize=13)
axs[0, 1].tick_params(axis='x', labelsize=13)
axs[0, 1].tick_params(axis='y', labelsize=13)
outliers = df[(df['Su_kPa'] > df['Su_kPa'].quantile(0.75) + 1.5 * (df['Su_kPa'].quantile(0.75) -
                                                                   df['Su_kPa'].quantile(0.25)))]
print(outliers)

# 3. Su vs PI and LI scatter plot.-----------------------------------------------------------------------------------
axs[1, 0].scatter(df["PI"], df["Su_kPa"], c='blue', label='Su vs PI', alpha=0.7)
axs[1, 0].scatter(df["LI"], df["Su_kPa"], c='green', label='Su vs LI', alpha=0.7)
axs[1, 0].set_xlabel('Plasticity Index (PI) or Liquidity Index (LI)',fontsize=13)
axs[1, 0].set_ylabel('Su (kPa)',fontsize=13)
axs[1, 0].set_title("Su vs PI and LI",fontsize=13)
axs[1, 0].legend()
axs[1, 0].tick_params(axis='x', labelsize=13)
axs[1, 0].tick_params(axis='y', labelsize=13)
axs[1, 0].grid(False)

# 4. Su vs Depth colored by Clay Type.-------------------------------------------------------------------------------
for clay in df["Clay_Type"].unique():
    subset = df[df["Clay_Type"] == clay]
    axs[1, 1].scatter(subset["Su_kPa"], subset["Dept (m)"], label=clay)
axs[1, 1].invert_yaxis()
axs[1, 1].set_xlabel("Su (kPa)",fontsize=13)
axs[1, 1].set_ylabel("Depth (m)",fontsize=13)
axs[1, 1].set_title("Su vs Depth by Clay Type",fontsize=13)
axs[1, 1].legend()
axs[1, 1].tick_params(axis='x', labelsize=13)
axs[1, 1].tick_params(axis='y', labelsize=13)
axs[1, 1].grid(False)

plt.tight_layout()
plt.savefig('E:\Documents\Su_analysis.png', dpi=600)
plt.show()

#=================================================== END OF CODE =======================================================