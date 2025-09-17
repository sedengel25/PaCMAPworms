# app.R
library(shiny)
library(tidyverse)
library(DT)
library(here)
library(readr)
library(ggplot2)
library(plotly)
library(parallelDist)
library(scales)
library(dbscan)
options(warn = -1)

ui <- fluidPage(
  titlePanel("PaCMAPworms – Showcase Results"),
  sidebarLayout(
    sidebarPanel(
      width = 2,                             # << kleiner Sidebar-Streifen
      uiOutput("run"),
      hr(),
      helpText("Wähle im Tab 'Table' eine Zeile. Der Tab 'Plot' zeigt dann die zugehörige Datei aus dem gewählten Unterordner."),
      selectInput(                           # << Filter im Sidebar lassen (optional)
        "color_by", "Färbung nach:",
        choices = c("Predicted" = "pred_labels", "True" = "true_labels"),
        selected = "pred_labels"
      )
    ),
    mainPanel(
      width = 10,
      tabsetPanel(id = "tabs",
                  tabPanel("Table",
                           DTOutput("tbl"),
                           br(),
                           verbatimTextOutput("sel_info")
                  ),
                  tabPanel("Plot",
                           # zwei Plots NEBENEINANDER
                           fluidRow(
                             column(6,
                                    plotlyOutput("org_plot3d", height = "70vh")  # höhe relativ zur Viewport-Höhe
                             ),
                             column(6,
                                    plotlyOutput("emb_plot2d", height = "70vh")
                             )
                           ),
                           fluidRow(
                             column(6,
                                    plotlyOutput("org_dist_density", height = "70vh")  # höhe relativ zur Viewport-Höhe
                             ),
                             column(6,
                                    plotlyOutput("emb_dist_density", height = "70vh")
                             )
                           )
                  )
      )
    )
  )
)


server <- function(input, output, session) {
  

  runs.avail <- reactive({
    dir_path <- here("showcase", "runs")
    if (!dir.exists(dir_path)) return(character(0))
    list.dirs(dir_path, full.names = FALSE, recursive = FALSE)
  })
  
  output$run <- renderUI({
    runs <- runs.avail()
    selectInput(
      "run", "Run (aus showcase/runs):",
      choices = runs, selected = if (length(runs)) runs[[1]] else NULL
    )
  })
  

  df.res <- reactive({
    req(input$run)
    csv_path <- here("results", paste0("results_", input$run, ".csv"))
    validate(need(file.exists(csv_path), paste0("Datei nicht gefunden: ", csv_path)))
    readr::read_csv(csv_path, show_col_types = FALSE)
  })

  # transform result table 
  df.view <- reactive({
    df <- df.res() %>%
      select(-any_of(c("DBCV_orig","DBCV_embedded_m", "DBCV_embedded_e"))) %>%
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
    base.dir <- here("showcase", "runs", run)
    df.gini <- readr::read_delim(here(base.dir, "gini.txt"), col_names = TRUE, delim = " ")
    df.cvi <- readr::read_delim(here(base.dir, "cvi.txt"), col_names = TRUE, delim = " ")
    print(head(df.gini))
    df <- df %>%
      left_join(df.gini, by = c("file" = "file"))
    df <- df %>%
      left_join(df.cvi, by = c("file" = "file"))
    df
  })
  
  # display result table
  output$tbl <- renderDT({
    datatable(
      df.view(),
      rownames = FALSE,
      filter = "top",
      selection = "single",
      options = list(pageLength = 25, scrollX = TRUE)
    )
  })
  

  selected_row <- reactive({
    req(input$tbl_rows_selected)
    df.view()[input$tbl_rows_selected, , drop = FALSE]
  })
  
  file.id <- reactive({
    as.character(selected_row()$file[1])
  })
  
  dimred_method <- reactive({
    as.character(selected_row()$dimred_method[1])
  })
  
  rep_id <- reactive({
    as.character(selected_row()$rep[1])
  })
  
  load.org.data <- reactive({
    run <- req(input$run)
    here("showcase", "runs", run, "org", paste0(file.id(), "_3d.txt"))
  })
  
  
  load.emb.data <- reactive({
    run <- req(input$run)
    base.dir <- here("showcase", "runs", run, "emb")
    file <- here(base.dir, paste0(file.id(), 
                                  "_", 
                                  dimred_method(), 
                                  "_", 
                                  rep_id(), 
                                  "_2d_emb.txt"))
    file
  })
  
  
  load.org.pred.labels <- reactive({
    file.id <- file.id()
    run <- req(input$run)
    base.dir <- here("showcase", "runs", run)
    labels <- here(base.dir, "pred_labels_org", paste0(file.id, "_pred_labels.txt"))
    labels
  })
  
  
  load.emb.pred.labels <- reactive({
    run <- req(input$run)
    base.dir <- here("showcase", "runs", run, "pred_labels_emb")
    labels <- here(base.dir, paste0(file.id(), 
                                  "_", 
                                  dimred_method(), 
                                  "_", 
                                  rep_id(), 
                                  "_2d_emb_pred_labels.txt"))
    labels
  })
  
  
  load.true.labels <- reactive({
    file.id <- file.id()
    run <- req(input$run)
    base.dir <- here("showcase", "runs", run)
    labels.true <- here(base.dir, "true_labels", paste0(file.id, "_labels.txt"))
    labels.true
  })


  
  
  output$emb_plot2d <- renderPlotly({
    df.char <- load.emb.data()
    pred.path <- load.emb.pred.labels()
    true.path <- load.true.labels()
    
    df <- readr::read_table(df.char, col_names = FALSE)
    df$pred_labels <- readr::read_table(pred.path, col_names = FALSE) %>% pull()
    df$true_labels <- readr::read_table(true.path, col_names = FALSE) %>% pull()
    df <- df %>% mutate(id = row_number())
    
    df$color_key <- as.factor(df[[input$color_by]])
    #df$dimred_key <- as.factor(df[[input$dimred]])

    picked <- sel_keys()

    
    if (length(picked) == 0) {
      # Normal: färben nach Labels
      plot_ly(
        data = df,
        x = ~X1, y = ~X2,
        type = "scatter", mode = "markers",
        color = ~color_key, colors = "Set1",
        key = ~id, source = "emb_select",
        marker = list(size = 2, opacity = 0.7)
      )
    } else {
      # Auswahl vorhanden: rot vs. hellgrau
      df <- df %>%
        mutate(
          is_sel = id %in% picked,
          col    = ifelse(is_sel, "red", "lightgray"),
          sz     = ifelse(is_sel, 4, 2),
          op     = ifelse(is_sel, 0.3, 0.05)
        )
      
      plot_ly(
        data = df,
        x = ~X1, y = ~X2,
        type = "scatter", mode = "markers",
        color = ~I(col),
        key = ~id, source = "emb_select",
        marker = list(size = ~sz, opacity = ~op),
        showlegend = FALSE
      )
    }
  })
  
  
  sel_keys <- reactive({
    s <- event_data("plotly_selected", source = "emb_select")
    if (is.null(s) || nrow(s) == 0) integer(0) else as.integer(s$key)
  })
  
  
  output$org_plot3d <- renderPlotly({
    df.char <- load.org.data()
    pred.path <- load.org.pred.labels()
    true.path <- load.true.labels()
    
    df <- readr::read_table(df.char, col_names = FALSE)

    k <- 100
    knn.d <- kNN(x = as.matrix(df), k = k)
    r.k   <- knn.d$dist[, k] + 1e-12              
    vol3  <- (4/3) * pi * (r.k^3) 
    density.knn <- k / vol3                    
    df <- df %>% mutate(density = density.knn)
    df <- df %>% mutate(id = row_number())
    df$pred_labels <- readr::read_table(pred.path, col_names = FALSE) %>% pull()
    df$true_labels <- readr::read_table(true.path, col_names = FALSE) %>% pull()
    
    df$color_key <- as.factor(df[[input$color_by]])
    
    picked <- sel_keys()
    if (length(picked) == 0) {
      plot_ly(
        data = df,
        x = ~X1, y = ~X2, z = ~X3,
        type = "scatter3d", mode = "markers",
        color = ~color_key, colors = "Set1",
        marker = list(size = 2, opacity = 0.7)
      ) %>%
        layout(legend = list(title = list(
          text = if (input$color_by == "pred_labels") "Predicted" else "True"
        )))
    } else {
      df <- df %>%
        mutate(
          is_sel = id %in% picked,
          col    = ifelse(is_sel, "red", "lightgray"),
          sz     = ifelse(is_sel, 4, 2),
          op     = ifelse(is_sel, 0.3, 0.05)
        )
      
      plot_ly(
        data = df,
        x = ~X1, y = ~X2, z = ~X3,
        type = "scatter3d", mode = "markers",
        color = ~I(col),                        # explizite Farben verwenden
        marker = list(size = ~sz, opacity = ~op),
        showlegend = FALSE
      )
    }
    
  })
  
  # output$org_dist_density <- renderPlotly({
  #   df <- readr::read_table(load.org.data(), col_names = FALSE)
  #   X  <- df %>% select(X1, X2, X3) %>% as.matrix()
  #   d  <- parallelDist::parDist(X, method = "euclidean") %>% as.vector()
  #   d <- scales::rescale(d, to = c(0, 1)) 
  #   g <- tibble(distance = d) %>%
  #     ggplot(aes(distance)) + geom_density() +
  #     labs(title = "Original 3D – Distanzdichte", x = "Distanz", y = "Dichte") +
  #     theme_minimal()
  #   ggplotly(g)
  # })
  # 
  # output$emb_dist_density <- renderPlotly({
  #   df <- readr::read_table(load.emb.data(), col_names = FALSE)
  #   X  <- df %>% select(X1, X2) %>% as.matrix()
  #   d  <- parallelDist::parDist(X, method = "euclidean") %>% as.vector()
  #   d <- scales::rescale(d, to = c(0, 1)) 
  #   
  #   g <- tibble(distance = d) %>%
  #     ggplot(aes(distance)) + geom_density() +
  #     labs(title = paste0(dimred_method(), " 2D – Distanzdichte"), x = "Distanz", y = "Dichte") +
  #     theme_minimal()
  #   ggplotly(g)
  # })
  
  

  observeEvent(input$tbl_rows_selected, {
    if (length(input$tbl_rows_selected) == 1) {
      updateTabsetPanel(session, "tabs", selected = "Plot")
    }
  })
}

shinyApp(ui, server)