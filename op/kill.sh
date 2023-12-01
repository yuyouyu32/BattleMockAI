ps aux | grep preprocess |  awk '{print $2}' | xargs kill -9
ps aux | grep train |  awk '{print $2}' | xargs kill -9
ps aux | grep server |  awk '{print $2}' | xargs kill -9

