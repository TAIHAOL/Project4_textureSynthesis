rm -rf outdir
mkdir outdir

INPUT_FILES=(apples.jpg cans.jpg cells.jpg fire.JPG jellybeans.jpg tomatoes.jpg water.JPG wood.JPG)
TEXTON_SIZE=(51 51 51 51 51 51 51 51)
OUTPUT_SIZE=(600 600 600 600 600 600 600 600)
OUTPUT_FILES=(apples.jpg cans.jpg cells.jpg fire.jpg jellybeans.jpg tomatoes.jpg water.jpg wood.jpg)

for ((i = 0; i < ${#INPUT_FILES[@]}; ++i)); 
do
    echo python main.py -input ./sampleTextures/${INPUT_FILES[$i]} -texton-size ${TEXTON_SIZE[$i]} -output-size ${OUTPUT_SIZE[$i]} -output ./outdir/${OUTPUT_FILES[$i]}
    python -u main.py -input ./sampleTextures/${INPUT_FILES[$i]} -texton-size ${TEXTON_SIZE[$i]} -output-size ${OUTPUT_SIZE[$i]} -output ./outdir/${OUTPUT_FILES[$i]}
done







