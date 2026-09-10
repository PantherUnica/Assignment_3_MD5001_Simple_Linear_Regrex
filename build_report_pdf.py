"""Builds Assignment1_Report.pdf from the content in Assignment1_Report.md, with plots embedded."""
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, Preformatted
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="H1", fontSize=18, leading=22, spaceAfter=12, fontName="Helvetica-Bold"))
styles.add(ParagraphStyle(name="H2", fontSize=13, leading=16, spaceBefore=14, spaceAfter=8, fontName="Helvetica-Bold"))
styles.add(ParagraphStyle(name="Meta", fontSize=9.5, leading=13, textColor=colors.HexColor("#444444")))
styles.add(ParagraphStyle(name="Body", fontSize=10.2, leading=14.5, spaceAfter=6, alignment=TA_LEFT))
styles.add(ParagraphStyle(name="ReportBullet", fontSize=10.2, leading=14.5, leftIndent=14, spaceAfter=6))
styles.add(ParagraphStyle(name="ReportCode", fontName="Courier", fontSize=8, leading=10.5,
                           backColor=colors.HexColor("#f5f5f5"), borderPadding=6))
styles.add(ParagraphStyle(name="ReportNote", fontSize=9.7, leading=13.5, leftIndent=10,
                           textColor=colors.HexColor("#333333"), borderColor=colors.HexColor("#999999"),
                           borderWidth=0.6, borderPadding=8, backColor=colors.HexColor("#fafafa")))

doc = SimpleDocTemplate("Assignment1_Report.pdf", pagesize=LETTER,
                         topMargin=0.7*inch, bottomMargin=0.7*inch,
                         leftMargin=0.75*inch, rightMargin=0.75*inch)

story = []

story.append(Paragraph("Assignment 1 — Simple Linear Regression: TV Advertising vs. Sales", styles["H1"]))
story.append(Paragraph(
    "<b>Author:</b> Aditya Raj &nbsp;|&nbsp; <b>Dataset:</b> tv_sales_data.csv "
    "(TV budget in $1,000s, Sales in 1,000s of units, n = 6)<br/>"
    "<b>Model:</b> Sales = &beta;0 + &beta;1 &middot; TV + &epsilon;<br/>"
    "<b>Method:</b> Coefficients fit via Gradient Descent; standard errors, R-squared, and p-values "
    "computed via the exact simple-linear-regression inference formulas applied to the fit's own "
    "residuals.",
    styles["Meta"]))
story.append(Spacer(1, 10))

# --- Section 1 ---
story.append(Paragraph("1. Code and Method", styles["H2"]))
story.append(Paragraph(
    "The model is fit by <b>Gradient Descent</b> instead of a closed-form formula: it starts "
    "both coefficients at 0, then repeatedly nudges them in the direction that reduces the Mean "
    "Squared Error (MSE), using a learning rate of &alpha; = 0.1 for 1000 iterations. The TV values "
    "are standardized before the loop purely for numerically stable steps, and the learned "
    "coefficients are converted back to the original ($1,000s) TV scale afterward.", styles["Body"]))

code1 = """x = df["TV"].values.astype(float)
y_vals = df["Sales"].values.astype(float)
n = len(x)

x_mean, x_std = x.mean(), x.std()
x_scaled = (x - x_mean) / x_std

b0, b1 = 0.0, 0.0
alpha = 0.1
n_iters = 1000

for i in range(n_iters):
    y_pred = b0 + b1 * x_scaled
    error = y_pred - y_vals
    grad_b0 = (2 / n) * np.sum(error)
    grad_b1 = (2 / n) * np.sum(error * x_scaled)
    b0 -= alpha * grad_b0
    b1 -= alpha * grad_b1

beta1 = b1 / x_std
beta0 = b0 - beta1 * x_mean"""
story.append(Preformatted(code1, styles["ReportCode"]))
story.append(Spacer(1, 6))

story.append(Paragraph(
    "Gradient Descent only minimizes MSE — it does not, by itself, produce standard errors, "
    "R-squared, or p-values. Those are calculated <b>directly from the fitted line's own "
    "residuals</b>, using the exact simple-linear-regression inference formulas:", styles["Body"]))

code2 = """from scipy import stats

y_fit = beta0 + beta1 * x
residuals = y_vals - y_fit

RSS = np.sum(residuals ** 2)
TSS = np.sum((y_vals - y_vals.mean()) ** 2)
df_resid = n - 2
sigma2 = RSS / df_resid
Sxx = np.sum((x - x_mean) ** 2)

se_beta0 = np.sqrt(sigma2 * (1 / n + x_mean ** 2 / Sxx))
se_beta1 = np.sqrt(sigma2 / Sxx)

t_beta0 = beta0 / se_beta0
t_beta1 = beta1 / se_beta1
p_beta0 = 2 * (1 - stats.t.cdf(abs(t_beta0), df_resid))
p_beta1 = 2 * (1 - stats.t.cdf(abs(t_beta1), df_resid))

r_squared = 1 - RSS / TSS"""
story.append(Preformatted(code2, styles["ReportCode"]))
story.append(Spacer(1, 6))

formula_data = [
    ["Quantity", "Formula"],
    ["Residual variance", "sigma^2 = RSS / (n - 2)"],
    ["SE(beta1)", "sqrt( sigma^2 / Sxx ),  Sxx = sum(xi - x_mean)^2"],
    ["SE(beta0)", "sqrt( sigma^2 * (1/n + x_mean^2/Sxx) )"],
    ["t-statistic", "beta_hat / SE(beta_hat), t-dist with n-2 df"],
    ["R-squared", "1 - RSS/TSS"],
]
t = Table(formula_data, colWidths=[1.6*inch, 4.6*inch])
t.setStyle(TableStyle([
    ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#2563eb")),
    ("TEXTCOLOR", (0,0), (-1,0), colors.white),
    ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
    ("FONTSIZE", (0,0), (-1,-1), 9),
    ("GRID", (0,0), (-1,-1), 0.5, colors.HexColor("#cccccc")),
    ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
    ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#f7f7f7")]),
    ("TOPPADDING", (0,0), (-1,-1), 5),
    ("BOTTOMPADDING", (0,0), (-1,-1), 5),
]))
story.append(t)
story.append(Spacer(1, 12))

# --- Section 2 ---
story.append(Paragraph("2. Results", styles["H2"]))
results_data = [
    ["Quantity", "Value"],
    ["beta0 (intercept)", "3.6000"],
    ["beta1 (slope, TV)", "0.5657"],
    ["SE(beta0)", "0.5674"],
    ["SE(beta1)", "0.0291"],
    ["t-statistic (beta0)", "6.345"],
    ["t-statistic (beta1)", "19.415"],
    ["p-value (intercept)", "0.003160"],
    ["p-value (slope)", "4.15 x 10^-5"],
    ["R-squared", "0.9895"],
    ["Degrees of freedom", "4 (n = 6, 2 parameters)"],
]
t2 = Table(results_data, colWidths=[2.6*inch, 3.6*inch])
t2.setStyle(TableStyle([
    ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#059669")),
    ("TEXTCOLOR", (0,0), (-1,0), colors.white),
    ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
    ("FONTSIZE", (0,0), (-1,-1), 9.5),
    ("GRID", (0,0), (-1,-1), 0.5, colors.HexColor("#cccccc")),
    ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
    ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#f7f7f7")]),
    ("TOPPADDING", (0,0), (-1,-1), 5),
    ("BOTTOMPADDING", (0,0), (-1,-1), 5),
]))
story.append(t2)
story.append(Spacer(1, 8))
story.append(Paragraph("Fitted equation: <b>Sales = 3.60 + 0.5657 x TV</b>", styles["Body"]))
story.append(Spacer(1, 10))

story.append(Paragraph("Fit and residuals:", styles["Body"]))
story.append(Image("regression_plot.png", width=6.3*inch, height=6.3*inch*(5/12)))
story.append(Spacer(1, 8))
story.append(Paragraph("Gradient Descent cost convergence:", styles["Body"]))
story.append(Image("gradient_descent_plot.png", width=4.2*inch, height=4.2*inch*(5/6)))
story.append(Paragraph(
    "The MSE falls smoothly toward its minimum over 1,000 iterations, confirming the model "
    "converged to a stable fit rather than stopping early or diverging.", styles["Body"]))
story.append(Spacer(1, 10))

# --- Section 3 ---
story.append(Paragraph("3. Interpretation — Plain Language", styles["H2"]))
plain_points = [
    "<b>What beta1 means in everyday terms:</b> beta1 = 0.5657 is a positive number, so spending "
    "more on TV ads goes with more sales. Since TV is measured in $1,000s and Sales in 1,000s of "
    "units, this means: for every extra $1,000 put into TV advertising, sales go up by roughly "
    "566 units, on average.",
    "<b>Is this a real effect, or could it be luck?</b> The p-value for the slope is 0.0000415 "
    "— a tiny number, way below the usual 0.05 cutoff people use to decide if something is "
    "&quot;statistically significant.&quot; In plain words: it's extremely unlikely we'd see a "
    "relationship this strong just by random chance if TV spend actually had no effect on sales. "
    "So yes, there's strong evidence TV advertising really does move sales.",
    "<b>How much of sales is &quot;explained&quot; by TV budget?</b> R-squared = 0.9895, meaning "
    "about 99% of the ups and downs in sales across these 6 data points can be accounted for just "
    "by knowing the TV budget. Only about 1% is left over as unexplained noise. That's an "
    "extremely tight fit — expected here because this is a small, clean practice dataset (only 6 "
    "points), not noisy real-world data.",
    "<b>Bottom line:</b> the data says TV advertising budget is a strong, reliable, positive "
    "driver of sales in this dataset, and a simple straight-line model captures almost all of "
    "the pattern.",
]
for p in plain_points:
    story.append(Paragraph("&bull; " + p, styles["ReportBullet"]))
story.append(Spacer(1, 8))

# --- Section 4 ---
story.append(Paragraph("4. Interpretation — Professional / Technical Language", styles["H2"]))
prof_points = [
    "<b>Sign and magnitude of beta1:</b> The estimated slope, beta1-hat = 0.5657, is positive and "
    "indicates a positive linear association between TV advertising expenditure and sales volume. "
    "Holding the linear specification fixed, each additional $1,000 unit of TV budget is "
    "associated with an expected increase of approximately 0.5657 thousand units (&asymp;566 "
    "units) in sales.",
    "<b>Statistical significance:</b> Under H0: beta1 = 0 versus H1: beta1 &ne; 0, the test "
    "statistic t = beta1-hat / SE(beta1-hat) = 19.415 on n-2 = 4 degrees of freedom yields "
    "p &asymp; 4.15 x 10^-5, far below conventional significance thresholds (&alpha; = 0.05, "
    "0.01). H0 is rejected, providing strong statistical evidence of a non-zero linear "
    "relationship between TV advertising and sales. The intercept is likewise statistically "
    "significant (beta0-hat = 3.60, SE = 0.5674, t = 6.345, p &asymp; 0.0032), though its "
    "interpretation as &quot;expected sales at zero TV spend&quot; is a mathematical "
    "extrapolation rather than a directly observed operating condition, since the sample range "
    "of TV spend is $5,000-$30,000.",
    "<b>Proportion of variance explained:</b> R-squared = 0.9895 indicates that approximately "
    "98.95% of the total variability in sales (TSS) is explained by the fitted linear "
    "relationship with TV advertising budget, leaving only &asymp;1.05% attributable to residual "
    "(unexplained) variation. Such a high R-squared is atypical for real marketing data and "
    "reflects the small sample size (n = 6) and the near-linear, low-noise construction of this "
    "illustrative dataset rather than a general claim about TV advertising's explanatory power "
    "in practice.",
    "<b>Overall conclusion:</b> Within this sample, TV advertising budget is a statistically "
    "significant and practically strong positive predictor of sales, and the simple linear model "
    "accounts for nearly all of the observed variance in the response.",
]
for p in prof_points:
    story.append(Paragraph("&bull; " + p, styles["ReportBullet"]))

doc.build(story)
print("Wrote Assignment1_Report.pdf")
