array=(12 24 36)
for i in "${array[@]}"
do
        for j in "${array[@]}"
        do
                for k in "${array[@]}"
                do
                        qsub test.sh -F "$i $j $k"
                done
        done
done
