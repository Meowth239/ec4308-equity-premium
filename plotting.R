plot_signal_timing <- function(yy, pred, labels = c(2000, 2005, 2010, 2015, 2020), at=c(8, 68, 128, 188, 247))  {# Calculate strategy returns
  actual_returns =  yy
  strategy = sign(pred)
  strategy_returns <- actual_returns * strategy
  
  # Create time index
  time_index <- 1:length(actual_returns)
  
  # Create the plot
  par(mar = c(5, 4, 4, 4))  # Adjust margins for secondary axis
  
  # Plot actual returns as bars
  barplot(actual_returns, 
          col = ifelse(actual_returns >= 0, "lightblue", "lightcoral"),
          border = NA,
          names.arg = "",
          xlab = "Period",
          ylab = "Actual Returns",
          main = "Signal Timing Analysis: Strategy Positions vs Actual Returns")
  
  
  
  # Add background shading for strategy positions
  plot(time_index, actual_returns, 
       type = "h",  # histogram-like vertical lines
       col = ifelse(actual_returns >= 0, "darkgreen", "darkred"),
       lwd = 2,
       xlab = "Period",
       ylab = "Actual Returns",
       main = "Signal Timing Analysis: Strategy Positions vs Actual Returns", xaxt = 'n')
  
  # Add horizontal line at zero
  abline(h = 0, col = "black", lwd = 1)
  
  # Add strategy signals as background shading
  # We'll add rectangles for when strategy = -1
  short_periods <- which(strategy == -1)
  for (i in short_periods) {
    rect(i - 0.5, par("usr")[3], i + 0.5, par("usr")[4], 
         col = rgb(1, 0, 0, 0.1), border = NA)  # Light red shading
  }
  
  # Add legend
  legend("topright", 
         legend = c("Positive Return", "Negative Return", "Short Position (-1)"),
         col = c("darkgreen", "darkred", rgb(1, 0, 0, 0.1)),
         lwd = c(2, 2, 10),
         bty = "n")
  axis(side = 1, at=c(8, 68, 128, 188, 247), labels = c(2000, 2005, 2010, 2015, 2020))
}