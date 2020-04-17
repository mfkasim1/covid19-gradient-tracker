export imgpath=/mnt/c/Users/firma/Documents/Projects/Git/mfkasim91.github.io/assets/idcovid19-daily
for fname in "id_new_cases" "idprov_jakarta_new_cases" \
    "idprov_jabar_new_cases" "idprov_jatim_new_cases" "idprov_sulsel_new_cases"
do
  echo ${fname}
  python models.py ${fname} --jit --savefig ${imgpath}/${fname}.png
done
