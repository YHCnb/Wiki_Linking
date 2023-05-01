
zh_extractor_path="zhwiki/zh_extractor"
zh_wiki_path="zhwiki/zh-wiki"

if [[ ! -f zh_wiki_path ]]; then
  c=0
  for small_file in $(find $zh_extractor_path -type f);
  do
    cat ${small_file} >> $zh_wiki_path
    c=$((c + 1))
    echo "Processed $c files"
  done
  echo "Mark-up is removed"
else
  echo "$zh_wiki_path already created"
fi