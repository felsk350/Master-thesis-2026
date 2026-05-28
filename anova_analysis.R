# ============================================================
# Two-Way ANOVA Analysis - Curve Crease Folding Study
# Factors: Crease Geometry (A, B, C) x Thickness (1.0, 1.5 mm)
# ============================================================

# Install required packages (run once)
# install.packages("car")
# install.packages("ggplot2")
# install.packages("dplyr")

library(car)
library(ggplot2)
library(dplyr)

# ============================================================
# 1. BUILD DATA
# ============================================================
df <- data.frame(
  sheet = c(
    "A_1_1","A_1_2","A_1_3","A_1_4","A_1_5",
    "A_1.5_1","A_1.5_2R","A_1.5_3","A_1.5_4","A_1.5_5",
    "B_1_1R","B_1_2","B_1_3","b_1_4","B_1_5",
    "B_1.5_1","B_1.5_2","B_1.5_3","B_1.5_4","B_1.5_5R",
    "C_1_1","C_1_2","C_1_3","C_1_4R","C_1_5R",
    "C_1.5_1R","C_1.5_2","C_1.5_3","C_1.5_4","C_1.5_5"
  ),
  
  geometry = factor(c(
    rep("A", 10), rep("B", 10), rep("C", 10)
  )),
  
  thickness = factor(c(
    rep("1.0", 5), rep("1.5", 5),
    rep("1.0", 5), rep("1.5", 5),
    rep("1.0", 5), rep("1.5", 5)
  ), levels = c("1.0", "1.5"),
  labels = c("1.0 mm", "1.5 mm")),
  
  # RMS deviation from group mean (mm)
  rms_vs_mean = c(
    0.1692, 0.2304, 0.2396, 0.0527, 0.0974,
    0.2069, 0.1094, 0.1608, 0.1108, 0.1919,
    0.0980, 0.1334, 0.1236, 0.0656, 0.1051,
    0.4636, 0.4248, 0.3851, 0.5637, 0.1075,
    0.0747, 0.0964, 0.0512, 0.0814, 0.1042,
    0.1468, 0.6333, 0.1644, 0.7827, 0.1191
  ),
  
  # RMS deviation from simulated plate (mm)
  rms_vs_sim = c(
    1.8704, 1.6063, 1.8920, 1.7040, 1.6841,
    4.2554, 4.4992, 4.4832, 4.3608, 4.5373,
    1.5170, 1.5902, 1.3704, 1.4702, 1.3934,
    4.9127, 4.9314, 5.4880, 5.6133, 5.1965,
    1.6964, 1.6115, 1.6843, 1.6090, 1.7188,
    1.4079, 1.0988, 1.4612, 2.0629, 1.5647
  ),
  
  # Mean absolute Gaussian curvature (1/mm2)
  K_abs_mean = c(
    9.26e-6, 9.04e-6, 7.57e-6, 7.61e-6, 9.92e-6,
    6.64e-6, 6.60e-6, 6.83e-6, 5.03e-6, 5.94e-6,
    1.872e-5, 1.846e-5, 2.078e-5, 1.801e-5, 1.810e-5,
    2.754e-5, 2.790e-5, 1.154e-5, 1.304e-5, 1.163e-5,
    2.951e-5, 2.968e-5, 3.085e-5, 3.133e-5, 2.970e-5,
    6.25e-6,  7.17e-6,  8.09e-6,  7.51e-6,  6.85e-6
  ),
  
  # Mean absolute K difference vs simulated (1/mm2)
  K_diff_abs_mean = c(
    1.805e-5, 1.989e-5, 1.803e-5, 1.757e-5, 1.805e-5,
    1.447e-5, 1.427e-5, 1.457e-5, 1.263e-5, 1.447e-5,
    1.872e-5, 1.486e-5, 1.962e-5, 1.759e-5, 2.078e-5,
    2.484e-5, 2.504e-5, 2.425e-5, 2.542e-5, 2.428e-5,
    3.266e-5, 3.355e-5, 3.531e-5, 3.615e-5, 3.155e-5,
    1.113e-5, 1.219e-5, 1.459e-5, 1.338e-5, 1.199e-5
  )
)

cat("Data built successfully:", nrow(df), "rows\n")
print(head(df))
cat("\n--- Summary ---\n")
print(summary(df))


# ============================================================
# 2. ANOVA FUNCTION
# ============================================================
run_anova <- function(response_name, response_label, data) {
  
  cat("\n", strrep("=", 60), "\n", sep = "")
  cat("  ANALYSIS:", response_label, "\n")
  cat(strrep("=", 60), "\n", sep = "")
  
  # Remove NA rows for this response
  data_clean <- data[!is.na(data[[response_name]]), ]
  
  formula <- as.formula(paste(response_name,
                              "~ geometry * thickness"))
  model <- aov(formula, data = data_clean)
  
  cat("\n--- ANOVA Table ---\n")
  print(summary(model))
  
  # Extract p-values
  anova_summary <- summary(model)[[1]]
  p_geom  <- anova_summary["geometry",           "Pr(>F)"]
  p_thick <- anova_summary["thickness",          "Pr(>F)"]
  p_int   <- anova_summary["geometry:thickness", "Pr(>F)"]
  
  cat("\n--- P-value Summary ---\n")
  cat("Geometry:    p =", round(p_geom,  4),
      ifelse(!is.na(p_geom)  && p_geom  < 0.05, " *SIGNIFICANT*", ""), "\n")
  cat("Thickness:   p =", round(p_thick, 4),
      ifelse(!is.na(p_thick) && p_thick < 0.05, " *SIGNIFICANT*", ""), "\n")
  cat("Interaction: p =", round(p_int,   4),
      ifelse(!is.na(p_int)   && p_int   < 0.05, " *SIGNIFICANT*", ""), "\n")
  
  if (!is.na(p_int) && p_int < 0.05) {
    cat("\nWARNING: Significant interaction - ",
        "interpret main effects with caution.\n")
  }
  
  # --- Shapiro-Wilk test ---
  cat("\n--- Shapiro-Wilk Test (Normality of Residuals) ---\n")
  sw <- shapiro.test(residuals(model))
  print(sw)
  cat(ifelse(sw$p.value < 0.05,
             "WARNING: Residuals may not be normally distributed\n",
             "OK: Normality assumption holds\n"))
  
  # --- Levene's test ---
  cat("\n--- Levene's Test (Homogeneity of Variance) ---\n")
  lev <- leveneTest(data_clean[[response_name]],
                    interaction(data_clean$geometry,
                                data_clean$thickness))
  print(lev)
  cat(ifelse(lev$`Pr(>F)`[1] < 0.05,
             "WARNING: Variances may not be equal across groups\n",
             "OK: Homogeneity of variance holds\n"))

  # --- Save assumption test results to CSV ---
  assumption_row <- data.frame(
    response = response_label,
    SW_W     = round(sw$statistic, 3),
    SW_p     = round(sw$p.value, 3),
    Levene_F = round(lev$`F value`[1], 2),
    Levene_p = round(lev$`Pr(>F)`[1], 3)
  )
  assumption_file <- "Shapiro_Wilk_Levene_results.csv"
  if (file.exists(assumption_file)) {
    write.table(assumption_row, assumption_file,
                sep = ",", row.names = FALSE,
                col.names = FALSE, append = TRUE)
  } else {
    write.csv(assumption_row, assumption_file, row.names = FALSE)
  }

  # --- Save ANOVA table to CSV ---
  anova_df         <- as.data.frame(summary(model)[[1]])
  anova_df$response <- response_label
  anova_df$source   <- rownames(anova_df)
  rownames(anova_df) <- NULL
  anova_file <- paste0("ANOVA_results_", response_name, ".csv")
  write.csv(anova_df, anova_file, row.names = FALSE)
  cat("Saved:", anova_file, "\n")
  
  # --- Q-Q plot ---
  png(paste0("qq_plot_", response_name, ".png"),
      width = 800, height = 600)
  qqnorm(residuals(model),
         main = paste("Q-Q Plot:", response_label))
  qqline(residuals(model), col = "red")
  dev.off()
  cat("Saved: qq_plot_", response_name, ".png\n", sep = "")
  
  # --- Tukey HSD for geometry (3 levels) ---
  if (!is.na(p_geom) && p_geom < 0.05) {
    cat("\n--- Tukey HSD Post-Hoc (Geometry) ---\n")
    tk <- TukeyHSD(model, "geometry")
    print(tk)
    png(paste0("tukey_", response_name, ".png"),
        width = 800, height = 600)
    plot(tk, main = paste("Tukey HSD:", response_label))
    dev.off()
    cat("Saved: tukey_", response_name, ".png\n", sep = "")
  } else {
    cat("\nGeometry not significant - no post-hoc test needed\n")
  }
  
  # --- Interaction plot ---
  png(paste0("interaction_", response_name, ".png"),
      width = 800, height = 600)
  interaction.plot(
    x.factor     = data_clean$geometry,
    trace.factor = data_clean$thickness,
    response     = data_clean[[response_name]],
    fun          = mean,
    type         = "b",
    col          = c("steelblue", "tomato"),
    lwd          = 2,
    legend       = TRUE,
    xlab         = "Crease Geometry",
    ylab         = paste("Mean", response_label),
    main         = paste("Interaction Plot:", response_label)
  )
  dev.off()
  cat("Saved: interaction_", response_name, ".png\n", sep = "")
  
  # --- Boxplot ---
  p <- ggplot(data_clean,
              aes(x = geometry,
                  y = .data[[response_name]],
                  fill = thickness)) +
    geom_boxplot(outlier.shape = 21, outlier.size = 2) +
    scale_fill_manual(values = c("steelblue", "tomato")) +
    labs(title = response_label,
         x     = "Crease Geometry",
         y     = response_label,
         fill  = "Thickness") +
    theme_minimal(base_size = 13)
  ggsave(paste0("boxplot_", response_name, ".png"), p,
         width = 8, height = 6, dpi = 200)
  cat("Saved: boxplot_", response_name, ".png\n", sep = "")
  
  return(model)
}


# ============================================================
# 3. RUN ANOVA FOR EACH RESPONSE VARIABLE
# ============================================================

m1 <- run_anova("rms_vs_mean",
                "RMS deviation from group mean (mm)", df)

m2 <- run_anova("rms_vs_sim",
                "RMS deviation from simulated plate (mm)", df)

m3 <- run_anova("K_abs_mean",
                "Mean absolute Gaussian curvature (1/mm2)", df)




# ============================================================
# 4. GROUP-LEVEL SUMMARY TABLE
# ============================================================
cat("\n", strrep("=", 60), "\n", sep = "")
cat("  GROUP-LEVEL MEANS\n")
cat(strrep("=", 60), "\n", sep = "")

summary_table <- df %>%
  group_by(geometry, thickness) %>%
  summarise(
    n               = n(),
    rms_vs_mean_mean = round(mean(rms_vs_mean, na.rm = TRUE), 4),
    rms_vs_mean_sd   = round(sd(rms_vs_mean,   na.rm = TRUE), 4),
    rms_vs_sim_mean  = round(mean(rms_vs_sim,  na.rm = TRUE), 4),
    rms_vs_sim_sd    = round(sd(rms_vs_sim,    na.rm = TRUE), 4),
    K_abs_mean_mean  = round(mean(K_abs_mean,  na.rm = TRUE), 8),
    K_diff_mean      = round(mean(K_diff_abs_mean, na.rm = TRUE), 8),
    .groups = "drop"
  )

print(summary_table)
write.csv(summary_table, "summary_table.csv", row.names = FALSE)
cat("\nSaved: summary_table.csv\n")
cat("\nAll done.\n")