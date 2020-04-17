---
layout: page
short_title: COVID-19 (ID)
title:
permalink: /idcovid19/
---

## Apakah kurva penyebaran telah turun?

Sumber data: [KawalCOVID19](https://kawalcovid19.id/)
{% for place in places %}
#### {{ place['name'] }}

Hasil: {{ place['flatcurve_result'] }}

<img title="{{ place['name'] }}" src="{{ '{{' }} site.baseurl {{ '}}' }}/assets/idcovid19-daily/{{ place['dataid'] }}.png" width="100%"/>
{% endfor %}
