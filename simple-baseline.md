For our simple-baseline, we randomly assigned a label/topic from 0 to n.

Sample execution:

python simple-baseline.py --input 'GenieMessagesTrain.csv' --output 'test_results.csv' --n_topics 5

Output is a csv with the following format (header and first row shown):

Combined.messages.to.Genie_ALL,Topic
Hi Genie PROPERNAME.Casebolt is nice its just i owe...,3

Silhouette Score on Test Data = -0.03394041753661137

This is about what you would expect from a random baseline, right in the middle of the range from -1 to 1.