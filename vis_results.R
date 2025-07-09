library(tidyverse)
library(here)
run <- "20250708_1538"
run.path <- here("runs", run, "data", "output")
df <- read_csv(paste0("results_", run, ".csv"))



df <- df %>%
  filter(dimred_method %in% c("PaCMAP", "tSNE"))
df <- df %>%
  select(-c(DBCV_orig, DBCV_embedded_m, DBCV_embedded_e))
unique(df$noise_mult)
df <- df %>%
  filter(noise_mult == 0)
summary(df)
str(df)
head(df %>% as.data.frame())

# df <- df %>%
#   mutate(
#     input_dim = str_extract(file, "(?<=_)\\d+(?=d\\.txt)") %>% as.integer()
#   )

df <- df %>%
  mutate(diff = ARI_embedded - ARI_orig)

df %>%
  filter(file == "02683_0nm_02682run_3d.txt")
df %>%
  arrange(diff) %>% as.data.frame() %>% head(10)
df %>%
  arrange(desc(diff)) %>% as.data.frame() %>% head(10)
# id <- 2567
# run <- id - 1
# file <- sprintf("%05d_0nm_%05drun", id, run)
# df.emb <- read_delim(here(run.path, paste0(file, "_2d_emb.txt")), delim = " ", col_names = FALSE) 
# labels.true <- read_lines(here(run.path, paste0(file, "_labels.txt")))
# labels.pred.org <- read_lines(here(run.path, paste0(file, "_pred_labels.txt")))
# labels.pred.emb <- read_lines(here(run.path, paste0(file, "_2d_emb_pred_labels.txt")))
# 
# df.emb <- df.emb %>%
#   mutate(label_truth = labels.true,
#          label_pred_org = labels.pred.org,
#          label_pred_emb = labels.pred.emb)
# 
# ggplot(data = df.emb, mapping = aes(x = X1, y = X2, color = label_pred_emb)) +
#   geom_point() +
#   theme_bw()

df.file <- df %>%
  select(
    file,
    dimred_method,
    ARI_orig,
    ARI_embedded
  ) %>%
  pivot_longer(
    cols = -c(file, dimred_method),
    names_to = "metric",
    values_to = "value"
  )


df.file %>%
  ggplot(aes(x = metric, y = value, fill = metric)) +
  geom_boxplot(outlier.size = 0.3, alpha = 0.7) +
  facet_wrap(~dimred_method) +
  labs(
    x = NULL,
    y = NULL
  ) +
  theme_minimal(base_size = 14) +
  theme(
    legend.position = "none",
    plot.title = element_text(hjust = 0.5),
    axis.text.x = element_text(angle = 45, hjust = 1)
  )





df.dim <- df %>%
  select(
    input_dim,
    target_dim,
    dimred_method,
    ARI_orig,
    ARI_embedded
  ) %>%
  pivot_longer(
    cols = -c(dimred_method, target_dim, input_dim),
    names_to = "metric",
    values_to = "value"
  )

df.dim %>%
  ggplot(aes(x = metric, y = value, fill = metric)) +
  geom_boxplot(outlier.size = 0.3, alpha = 0.7) +
  facet_grid(rows = vars(input_dim, target_dim), cols = vars(dimred_method), scales = "fixed") +
  labs(
    x = NULL,
    y = NULL
  ) +
  theme_minimal(base_size = 14) +
  theme(
    legend.position = "none",
    plot.title = element_text(hjust = 0.5),
    axis.text.x = element_text(angle = 45, hjust = 1)
  )


df.hdbscan <- df %>%
  select(
    min_cluster_size,
    min_samples,
    ARI_orig,
    ARI_embedded
  ) %>%
  pivot_longer(
    cols = -c(min_cluster_size, min_samples),
    names_to = "metric",
    values_to = "value"
  )


df.hdbscan %>%
  ggplot(aes(x = metric, y = value, fill = metric)) +
  geom_boxplot(outlier.size = 0.3, alpha = 0.7) +
  facet_wrap(~min_cluster_size) +
  labs(
    x = NULL,
    y = NULL
  ) +
  theme_minimal(base_size = 14) +
  theme(
    legend.position = "none",
    plot.title = element_text(hjust = 0.5),
    axis.text.x = element_text(angle = 45, hjust = 1)
  )


# df.long %>%
#   # filter(metric %in% c("ARI_embedded", "ARI_org")) %>%
#   arrange(desc(value)) %>%
#   print(n=150)
# 
# df %>%
#   arrange(desc(ARI_embedded)) %>%
#   select(ARI_embedded, file, min_cluster_size, min_samples) %>%
#   print(n=100)
# threshold <- 0.67
# df <- df %>%
#   mutate(flag_high_ARI_embedded = ARI_embedded > threshold)
# 
# num.vars <- df %>%
#   select(where(is.numeric)) %>%
#   select(-ARI_embedded) %>%  
#   names()
# 
# df.long <- df %>%
#   select(all_of(num.vars), 
#          flag_high_ARI_embedded, 
#          -c(n_clusters_orig, n_clusters_embedded, 
#             n_noise_orig, n_noise_embedded)) %>%
#   pivot_longer(
#     cols = -flag_high_ARI_embedded,
#     names_to = "variable",
#     values_to = "value"
#   )
# 
# df.long %>%
#   ggplot(aes(x = flag_high_ARI_embedded, y = value, fill = flag_high_ARI_embedded)) +
#   geom_boxplot(outlier.size = 0.3, alpha = 0.7) +
#   facet_wrap(~ variable, scales = "free_y") +
#   labs(
#     x = NULL,
#     y = NULL,
#     fill = paste0("ARI (embedded) > ", threshold)
#   ) +
#   theme_minimal(base_size = 13) +
#   theme(
#     plot.title = element_text(hjust = 0.5),
#     axis.text.x = element_text(angle = 0, hjust = 0.5)
#   )
# 
# 
# df.pacmap <- df %>%
#   filter(dimred_method == "PaCMAP") %>%
#   select(
#     n_neighbors,
#     ARI_orig,
#     DBCV_orig,
#     ARI_embedded,
#     DBCV_embedded_m
#   ) %>%
#   pivot_longer(
#     cols = -n_neighbors,
#     names_to = "metric",
#     values_to = "value"
#   )
# 
# df.pacmap %>%
#   ggplot(aes(
#     x = metric,
#     y = value,
#     fill = metric
#   )) +
#   geom_boxplot(
#     outlier.size = 0.3,
#     alpha = 0.7
#   ) +
#   facet_wrap(~ n_neighbors, scales = "free_y") +
#   labs(
#     x = NULL,
#     y = NULL,
#     fill = NULL,
#     title = "PaCMAP: Metrics by n_neighbors"
#   ) +
#   theme_minimal(base_size = 13) +
#   theme(
#     plot.title = element_text(hjust = 0.5),
#     axis.text.x = element_text(angle = 45, hjust = 1)
#   )

