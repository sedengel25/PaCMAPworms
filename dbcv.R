library(tidyverse)
library(here)
library(corrplot)
run <- "20250807_1214"
path <- here("showcase", "runs", run)

# files <- list.files(path = path, full.names = TRUE)
# 
# 
# raw <- map_chr(files, readr::read_lines)
# 
# df.cvi <- tibble(raw = raw) %>%
#   separate(raw, into = c("file", "sil", "dbcv"), sep = " ")
# 
# 
# df.cvi <- df.cvi %>%
#   mutate(sil = as.numeric(sil),
#          dbcv = as.numeric(dbcv),
#          file = str_replace(file, "(run).*", "\\1"))
# path.res <- here("results", paste0("results_", run, ".csv"))
# write_delim(df, here("showcase", "runs", run, "cvi.txt"), delim = " ")


df.cvi <- readr::read_delim(here(path, "cvi.txt"), col_names = TRUE, delim = " ")
summary(df.cvi)
df.res <- readr::read_csv(path.res, show_col_types = FALSE)
df.res <- df.res %>%
  mutate(
    diff = ARI_embedded - ARI_orig,
    file = str_replace(file, "(run).*", "\\1")
  ) %>%
  group_by(file, dimred_method) %>%
  mutate(
    diff_mean   = mean(diff, na.rm = TRUE),
    diff_median = median(diff, na.rm = TRUE)
  ) %>%
  ungroup()

str(df.res)

df.res <- df.res %>%
  left_join(df.cvi, by = c("file" = "file"))

summary(df.res)

df <- df.res %>%
  filter(dimred_method == "tSNE")


df <- df[,c("ARI_orig", "ARI_embedded", "dbcv")]
ggplot(df, aes(x = ARI_orig, y = ARI_embedded, color = dbcv)) +
  geom_point(alpha = 0.2) +
  theme_bw() +
  scale_color_viridis_c(option = "viridis") +
  coord_cartesian(xlim = c(0, 1), ylim = c(0, 1)) +
  geom_abline(slope = 1, intercept = 0, color = "red", linetype = "dashed")




df.res %>%
  arrange(desc(dbcv)) %>%
  as.data.frame()




df.plot <- df.res %>%
  mutate(
    ari_better = if_else(ARI_embedded > ARI_orig, "ARI embedded > ARI orig", "ARI embedded ≤ ARI orig"),
    dimred_method = factor(dimred_method, levels = c("PaCMAP", "TriMap", "UMAP", "tSNE"))
  )


df.plot$file %>% unique
# ---- Plot: Violin + Box, gefacettet nach DimRed ----
ggplot(df.plot, aes(x = ari_better, y = dbcv, fill = ari_better)) +
  # geom_violin(trim = TRUE, scale = "width") +
  geom_boxplot(width = 0.12, alpha = 0.8) +
  facet_wrap(~ dimred_method, ncol = 2) +
  labs(
    x = NULL, y = "dbcv"
  ) +
  guides(fill = "none") +
  theme_bw(base_size = 12) +
  theme(panel.spacing = unit(10, "pt"), axis.text.x = element_text(angle = 15, hjust = 1))


ggplot(df.res, mapping = (aes(x = dbcv))) +
  geom_density()
