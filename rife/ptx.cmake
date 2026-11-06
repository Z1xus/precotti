file(READ ${input} hex HEX)
string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1," hex "${hex}")
file(WRITE ${output} "static const char kernels_ptx[] = {${hex}0};\n")
