#!/bin/bash
curl -o input.txt \
  https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
echo "Downloaded $(wc -c < input.txt) bytes"