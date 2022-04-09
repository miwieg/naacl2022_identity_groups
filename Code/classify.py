
infile = "output.sample.txt" #ADJUST TO YOUR SETTINGS
outfile = "prediction.sample.txt" #ADJUST TO YOUR SETTINGS


def main():
    with open(infile) as f:
        with open(outfile, "w") as fw:
            is_first_line = True
            for line in f:
                line = line.strip()
                if is_first_line:
                    # the first line includes description of column information
                    is_first_line = False
                    fw.write(line + "\tPREDICTION\n")
                else:
                    columns = line.split("\t")
                    verb = columns[1]
                    aspect_prediction = columns[2]
                    perpetrator_label = columns[3]
                    non_conformist_view_label = columns[-1]
                    prediction = "OTHER"
                    if not(aspect_prediction == "EPISODIC"):
                        if perpetrator_label == "PERPETRATOR":
                            prediction = "ABUSE"
                        elif non_conformist_view_label == "NON-CONFORMISTIC":
                            prediction = "ABUSE"
                    if verb == "UNKNOWN_VERB":
                        # in the unlikely event that the verb could not be identified
                        # make sure that prediction is "OTHER" (i.e. default class)
                        prediction = "OTHER"
                    fw.write(line + "\t" + prediction + "\n")




if __name__ == '__main__':
    main()
