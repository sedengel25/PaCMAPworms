# app.R
library(shiny)
library(tidyverse)
library(DT)
library(here)
library(readr)
library(ggplot2)
library(plotly)


ui <- fluidPage(
  titlePanel("PaCMAPworms – Showcase Results"),
  sidebarLayout(
    sidebarPanel(
      uiOutput("run"),
      hr(),
      helpText("Wähle im Tab 'Table' eine Zeile. Der Tab 'Original' zeigt dann die zugehörige Datei aus dem gewählten Unterordner.")
    ),
    mainPanel(
      tabsetPanel(id = "tabs",
                  tabPanel("Table",
                           DTOutput("tbl"),
                           br(),
                           verbatimTextOutput("sel_info")
                  ),
                  tabPanel("Original",
                           plotlyOutput("org_plot3d", height = "600px")
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
      select(-any_of(c("DBCV_orig", "DBCV_embedded_m", "DBCV_embedded_e"))) %>%
      mutate(
        diff = ARI_embedded - ARI_orig,
        file = str_replace(file, "(run).*", "\\1")
      )
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
  
  
  file.id <- reactive({
    req(input$run)
    s <- input$tbl_rows_selected
    validate(need(length(s) == 1, "Bitte wähle eine Zeile in der Tabelle."))
    df <- df.view()
    row <- df[s, , drop = FALSE]
    
    validate(need("file" %in% names(row), "Spalte 'file' fehlt im results-CSV."))
    file.id <- as.character(row$file[1])
    file.id
  })
  

  


  # load_selected_file <- reactive({
  #   path <- resolved_path()
  #   print(path)
  #   df_try <- tryCatch(
  #     readr::read_table(path, show_col_types = FALSE, progress = FALSE),
  #     error = function(e) NULL
  #   )
  #   if (is.null(df_try)) {
  #     df_try <- tryCatch(
  #       readr::read_csv(path, show_col_types = FALSE, progress = FALSE),
  #       error = function(e) NULL
  #     )
  #   }
  #   validate(need(!is.null(df_try), paste0("Datei konnte nicht geparst werden: ", path)))
  #   
  #   df_try
  # })
  
  pick_three_numeric <- function(dat) {
    num_cols <- names(dat)[map_lgl(dat, is.numeric)]
    validate(need(length(num_cols) >= 3,
                  "Weniger als drei numerische Spalten gefunden – 3D-Plot nicht möglich."))
    num_cols[1:3]
  }
  
  output$org_plot3d <- renderPlotly({
    dat <- load_selected_file()
    xyz <- pick_three_numeric(dat)
    
    plot_ly(
      data = dat,
      x = ~.data[[xyz[1]]],
      y = ~.data[[xyz[2]]],
      z = ~.data[[xyz[3]]],
      type = "scatter3d",
      mode = "markers",
      marker = list(size = 2, opacity = 0.7)
    ) %>%
      layout(
        scene = list(
          xaxis = list(title = xyz[1]),
          yaxis = list(title = xyz[2]),
          zaxis = list(title = xyz[3]),
          aspectmode = "data"
          ),
          margin = list(l = 0, r = 0, b = 0, t = 0)
        )
  })
    
  
  # Optional: bei Auswahl in der Tabelle automatisch auf den Original-Tab springen
  observeEvent(input$tbl_rows_selected, {
    if (length(input$tbl_rows_selected) == 1) {
      updateTabsetPanel(session, "tabs", selected = "Original")
    }
  })
}

shinyApp(ui, server)
