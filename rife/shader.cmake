# a shader as a c array, ncnn compiles it when the plugin starts
file(READ ${input} hex HEX)
string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1," hex "${hex}")
get_filename_component(name ${input} NAME_WE)
file(WRITE ${output} "static const char ${name}_comp_data[] = {${hex}};\n")
