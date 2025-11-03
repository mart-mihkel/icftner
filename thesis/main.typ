#import "lib.typ": title-page, info-eng, info-est

#set par(justify: true)
#set page(margin: 25.4mm)
#set heading(numbering: "1.")
#set text(font: "New Computer Modern", size: 12pt)

#title-page(
  thesis-title: "Prefix Tuning Applicability In Language Models",
  author: "Mart-Mihkel Aun",
  supervisor: "Sven Laur, DSc",
  curriculum: "Data Science Curriculum",
  year: 2026,
)

#info-eng(
  thesis-title: "Prefix Tuning Applicability In Language Models",
  keywords: "LLM, fine-tuning",
  cercs: "P176",
)[#lorem(50)]

#info-est(
  thesis-title: "Prefix Tuning Applicability In Language Models",
  keywords: "LLM, fine-tuning",
  cercs: "P176",
)[#lorem(50)]

#pagebreak()

#outline()
#pagebreak()

= Introduction

$ h_i = "LM"_phi.alt (z_i, h_(lt i)) $

= Related Work

== Context based parameter efficient fine-tuning

Prefix tuning @li2021prefix, P-Tuning @liu2024gpt, Prompt tuning @lester2021power.

== Weight based parameter efficient fine-tuning

LoRA @hu2022lora.

== Language models

BERT @devlin2019bert, Llama @touvron2023llama.

Few-shot learning @brown2020language.

= Conclusion

#bibliography("ref.bib")

= Appendices

= License
