library(tidyverse)
library(here)

idx  <- 7542
idx2 <- idx - 1

dim.red <- "tSNE"
rep <- 3
?str_pad

idx_str  <- str_pad(idx,  5, pad = "0")
idx2_str <- str_pad(idx2, 5, pad = "0")

df.org <- read_delim(
  here("example_plots", paste0(idx_str, "_0nm_", idx2_str, "run_3d.txt")),
  col_names = FALSE,
  delim = " "
)

df.org.pred.labels <- read_delim(
  here("example_plots", paste0(idx_str, "_0nm_", idx2_str, "run_pred_labels.txt")),
  col_names = FALSE,
  delim = " "
)



df.emb <- read_delim(
  here("example_plots", paste0(idx_str, 
                               "_0nm_", 
                               idx2_str, 
                               "run_", 
                               dim.red,
                               "_",
                               rep,
                               "_2d_emb.txt")),
  col_names = FALSE,
  delim = " "
)

df.emb.pred.labels <- read_csv(
  here("example_plots", paste0(idx_str, 
                               "_0nm_", 
                               idx2_str, 
                               "run_", 
                               dim.red,
                               "_",
                               rep,
                               "_2d_emb_pred_labels.txt")),
  col_names = FALSE
)


ggplot(data = df.emb, mapping = aes(x = X1, y = X2)) +
  geom_point()



