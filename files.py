from database import database_df
import os

for i in range(len(database_df)):
    if database_df['category'][i].lower() == "business":
        path = "datasets/{}/".format(database_df["category"][i])
        filename = "{}.txt".format(i)
        path_final = os.path.join(path, filename)
        for content in database_df[database_df["category"] == "{}".format(database_df['category'][i])]:
            with open(path_final, "w") as f:
                f.write(database_df["title"][i]+"\n"+database_df["content"][i])

    elif database_df['category'][i].lower() == "entertainment":
        path = "datasets/{}/".format(database_df["category"][i])
        filename = "{}.txt".format(i)
        path_final = os.path.join(path, filename)
        for content in database_df[database_df["category"] == "{}".format(database_df['category'][i])]:
            with open(path_final, "w") as f:
                f.write(database_df["title"][i]+"\n"+database_df["content"][i])

    elif database_df['category'][i].lower() == "sports":
        path = "datasets/{}/".format(database_df["category"][i])
        filename = "{}.txt".format(i)
        path_final = os.path.join(path, filename)
        for content in database_df[database_df["category"] == "{}".format(database_df['category'][i])]:
            with open(path_final, "w") as f:
                f.write(database_df["title"][i]+"\n"+database_df["content"][i])

    elif database_df['category'][i].lower() == "politics":
        path = "datasets/{}/".format(database_df["category"][i])
        filename = "{}.txt".format(i)
        path_final = os.path.join(path, filename)
        for content in database_df[database_df["category"] == "{}".format(database_df['category'][i])]:
            with open(path_final, "w") as f:
                f.write(database_df["title"][i]+"\n"+database_df["content"][i])

    elif database_df['category'][i].lower() == "tech":
        path = "datasets/{}/".format(database_df["category"][i])
        filename = "{}.txt".format(i)
        path_final = os.path.join(path, filename)
        for content in database_df[database_df["category"] == "{}".format(database_df['category'][i])]:
            with open(path_final, "w") as f:
                f.write(database_df["title"][i]+"\n"+database_df["content"][i])
