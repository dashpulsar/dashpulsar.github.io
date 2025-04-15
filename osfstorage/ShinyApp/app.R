library("shiny")
library("tidyverse")
library("png")
load("Shiny_morpho.RData")
head_outline_png <-  readPNG('head_outline2.png')
###features to add
#toggle labels on/off
#higher rest head outline

ui<-navbarPage(title = "VTL Companion App",
  tabPanel(title="Figures",
  wellPanel(#grey box to visually bind output segment
    fluidRow(column(6,plotOutput(outputId="Scores", width="400px", height="400px")),
        column(6,plotOutput(outputId="Vocal_Tract", width="400px", height="400px"))
             )#end fluidrow
           ),#end panel
  wellPanel( #grey box to visually bind input segment
    fluidRow(
        column(4,     
             selectInput(inputId="PCX",label="x-axis", selected="PC1",choices=c("PC1","PC2","PC3","PC4","PC5")),
             selectInput(inputId="PCY",label="y-axis", selected="PC4",choices=c("PC1","PC2","PC3","PC4","PC5")),
             selectInput(inputId="Col",label="Colour", selected="Word",choices=c("Word","VTL"))#Pitch, ID also available
        ),#endcolumn  
      
        column(4, #first 2 inpytsliders
              sliderInput(inputId="PC1", min=-20, max=20, value=0,
                    label="PC1 score (Tongue Position: 40.7%)"),
              sliderInput(inputId="PC2", min=-20, max=20, value=0,
                    label="PC2 score (Vocal Tract Length: 34.0%)"),
              actionButton(inputId = "Reset_Scores",label="Reset Scores")
        ),#endcolumn 
          column(4, #the rest of the input sliders
             sliderInput(inputId="PC3", min=-20, max=20, value=0,
                    label="PC3 score (Body Size: 9.3%)"),
             sliderInput(inputId="PC4", min=-20, max=20, value=0,
                    label="PC4 score (Tongue Shape: 8.2%)"),
             sliderInput(inputId="PC5", min=-20, max=20, value=0,
                    label="PC5 score (Curvature: 3.5%)")
          )#endcolumn 
    )#end fluidrow
  )#end panel
  ),#end tab
  #new tab for legend
  tabPanel(title="Legend",
           h1("Figure Legend:"), 
           p({"We used functional principal components analysis to 
             study variation in vocal tract shapes. 
             Participants said the words BEAD and BARD 
             while shortening or lenghtening their vocal tracts.
             This companion app allows you to interactively explore the dimensions along which vocal tract shapes varied. 
             Adjust the sliders on the Figures tab to add or subtract any combination of principal components. 
             Component labels provide a subjective interpretation of each component and report the portion of variation accounted for by each.
             The scatterplot (top left) shows scores for each vocal tract image
             and a crosshair to highlight the currently selected combination of scores. 
             Use the dropdown menu to choose which scores are plotted.
             The shape plot (top right shows the 
             corresponding vocal tract shape (black) 
             as well the mean vocal tract shape (light grey)."}))
)#end fluidpage 

server<-function(input,output, session){ 
  
  observeEvent(input$Reset_Scores, {
    updateNumericInput(session, "PC1", value = 0)
    updateNumericInput(session, "PC2", value = 0)
    updateNumericInput(session, "PC3", value = 0)
    updateNumericInput(session, "PC4", value = 0)
    updateNumericInput(session, "PC5", value = 0)
    
  })
  
  #find x-y values from selected sliders
  scatterplot_X<-reactive({
       if(input$PCX=="PC1"){input$PC1}
  else if(input$PCX=="PC2"){input$PC2}
  else if(input$PCX=="PC3"){input$PC3}
  else if(input$PCX=="PC4"){input$PC4}
  else                     {input$PC5}
})
  
  scatterplot_Y<-reactive({
         if(input$PCY=="PC1"){input$PC1}
    else if(input$PCY=="PC2"){input$PC2}
    else if(input$PCY=="PC3"){input$PC3}
    else if(input$PCY=="PC4"){input$PC4}
    else                     {input$PC5}
  })

  output$Scores<-renderPlot({
    ggplot(data=Shiny_scores, aes_string(x=input$PCX, y=input$PCY, colour=input$Col, fill=input$Col))+ #aes_string because input$PCX contains a string
      geom_point(size=0.25)+
      geom_vline(xintercept=scatterplot_X())+
      geom_hline(yintercept=scatterplot_Y())+
      xlim(-25,25)+
      ylim(-25,25)+
      ggtitle("Component Scores")+
      guides(col = guide_legend(override.aes = list(shape = 15, size = 3)))+
      theme_light()
  })
  
  #replace base R with ggplot
  #something a bout getting these reactive objects into a dataframe seems hard
  
  
  Shiny_PC_reactive <- reactive({Shiny_PC %>% #math out estimated vocal tract shape
                              mutate(ycoords=ymean+
                                       ypc1*input$PC1+
                                       ypc2*input$PC2+
                                       ypc3*input$PC3+
                                       ypc4*input$PC4+
                                       ypc5*input$PC5) %>%
                              mutate(zcoords=zmean+
                                      zpc1*input$PC1+
                                      zpc2*input$PC2+
                                      zpc3*input$PC3+
                                      zpc4*input$PC4+
                                      zpc5*input$PC5) %>%
                            mutate(ycoords=ycoords-ycoords[1]) %>% #set origin to first value, upper lip
                            mutate(zcoords=zcoords-zcoords[1])     #set origin to first value, upper lip
    
  })
  
  output$Vocal_Tract<-renderPlot({
    
    #plots shapes
    ggplot(Shiny_PC_reactive(),aes(x=ymean, y=zmean))+
      geom_polygon(color="grey90", aes(fill="grey90"))+
      geom_polygon(aes(x=ycoords,
                       y=zcoords,
                       fill="black"))+
      geom_polygon(color="lightgrey", fill=NA, linetype=3)+
      
      #add text annotations to plot
      geom_text(aes(x=ycoords[1]-3,y=zcoords[1]+1, label="Lips"))+
      geom_text(aes(x=max(ycoords)+2.5,y=min(zcoords)-1, label="Larynx"))+
      geom_text(aes(x=ycoords[75],y=zcoords[75]-5, label="Tongue"))+
      
      #add head outline for context
      annotation_raster(head_outline_png, ymin =-29,ymax= 61,xmin = -15,xmax = 30) + 
      
      #style
      ylim(-25, 30)+
      xlim(-10, 45)+
      xlab("Y (pixels from upper lip)")+
      ylab("Z (pixels from upper lip)")+
      ggtitle("Vocal Tract Shapes")+
      scale_fill_identity(name=element_blank(),guide="legend",labels=c("Estimated Vocal Tract", "Mean Vocal Tract"))+
      guides(fill=guide_legend())+
      theme_light()+
      theme(legend.position = c(.80, .90),legend.box.background = element_rect(colour = "black"))
  })
}

shinyApp(ui=ui,server=server) #run it