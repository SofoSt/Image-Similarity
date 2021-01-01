import argparse
import sys
import hiplot as hip
import webbrowser

# create a parser in order to obtain the arguments
parser = argparse.ArgumentParser(description='Create a hi-plot')
# the oonly argument that we want is -d
parser.add_argument('-i', '--input', action='store', default=None,  metavar='', help='Relative path to the hiplot')
# parse the arguments
args = parser.parse_args()

# get the result of hiplot in an html page
html_str = hip.Experiment.from_csv(args.input).to_html()
# open a file to save the html containing the plot 
f_name = "../Results/hiplots/hiplot_result_csv.html"

f = open(f_name, "w")
f.write(html_str)
f.close()
# pop-up in google chrome
webbrowser.get('/usr/bin/google-chrome %s').open(f_name)

