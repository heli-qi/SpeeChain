def text2word_list(x: str):
    word_list = []
    for word in x.split():
        # no punctuation
        if word[0].isalpha() and word[-1].isalpha():
            word_list.append(word)
        # punctuation attached at the beginning
        elif not word[-1].isalpha():
            word_list.append(''.join(word[:-1]))
            word_list.append(word[-1])
        # punctuation attached at the end
        elif not word[0].isalpha():
            word_list.append(word[0])
            word_list.append(''.join(word[1:]))
        # punctuation attached at the beginning and the end
        else:
            word_list.append(word[0])
            word_list.append(''.join(word[1:-1]))
            word_list.append(word[-1])
    return word_list