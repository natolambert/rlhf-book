# Introduction

This is the first paragraph of the introduction chapter.
This is a test of citing [@lambert2023entangled].

## First: Images

This is the first subsection. Please, admire the gloriousnes of this seagull:

![A cool seagull.](images/seagull.png)

A bigger seagull:

![A cool big seagull.](images/seagull.png){ width=320px }

## Second: Tables

This is the second subsection.


Please, check [First: Images] subsection.

Please, check [this](#first-images) subsection.

| Index | Name |
| ----- | ---- |
| 0     | AAA  |
| 1     | BBB  |
| ...   | ...  |

Table: This is an example table.

## Third: Equations

Formula example: $\mu = \sum_{i=0}^{N} \frac{x_i}{N}$

Now, full size:

$$\mu = \sum_{i=0}^{N} \frac{x_i}{N}$$

And a code sample:

```rb
def hello_world
  puts "hello world!"
end

hello_world
```

Check these unicode characters: ǽß¢ð€đŋμ

## Fourth: Cross references

These cross references are disabled by default. To enable them, check the
_[Cross references](https://github.com/wikiti/pandoc-book-template#cross-references)_
section on the README.md file.

Here's a list of cross references:

- Check @fig:seagull.
- Check @tbl:table.
- Check @eq:equation.

![A cool seagull](images/seagull.png){#fig:seagull}

$$ y = mx + b $$ {#eq:equation}

| Index | Name |
| ----- | ---- |
| 0     | AAA  |
| 1     | BBB  |
| ...   | ...  |

Table: This is an example table. {#tbl:table}