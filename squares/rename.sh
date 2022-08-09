find . -name 'rook_*.png' -type f -exec sh -c '
for f; do
    mv "$f" "${f%/*}/${f##*/rook_}"
done' sh {} +
