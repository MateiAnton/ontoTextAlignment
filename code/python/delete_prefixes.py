# script that takes a ttl file and deletes all lines that start with a spcific string


# open file
with open('modified_original_data/Merged_GeoFault.ttl', 'r') as f:
    lines = f.readlines()

prefix_to_delete = "obo:IAO_0000115"
    

# delete lines that start with the prefix
lines = [line for line in lines if not line.lstrip().startswith(prefix_to_delete)]

# write to file
with open('modified_original_data/Merged_GeoFault_modified.ttl', 'w') as f:
    f.writelines(lines)
