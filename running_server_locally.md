#### run evaluations and gather results

In order to deploy our results locally skip to 'run streamlit'.

To deploy your own results, run evaluations on all of your desired models as specified in the Evaluate MRR section. Then gather all of the results to a dataframe and save it in a location of your choice. Note that the dynamic visualization is intended for bidirectional relations, so make sure to evaluate all couples in your langauge set. Currently, we only support evaluation on one of the 22 languages used in our experiments. To see your results, pass the gathered dataframe explicilty using '--df_path' when calling 'launch_interface.py', as explained next. 

#### run streamlit
Install requirements found in language-graph/visualization_tool dir using:
```
pip install -r requirements.txt
```
Then, in your terminal open the project root directory and run:
```
streamlit run visualization_tool/launch_interface.py --server.port=PORT 
```
Finally, visit the generate URL, printed in your console.
