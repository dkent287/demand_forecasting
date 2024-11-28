
test_fun <- function(df,alist){
  df$Values <- df$Values * 3
  alist <- list(1,2)
  l_final = list(df,alist)
  return(l_final)
}
