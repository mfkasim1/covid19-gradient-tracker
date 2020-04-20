---
layout: page
short_title:
title:
permalink: /idcovid19/
---

## Apakah kurva penyebaran telah turun?

Sumber data: [KawalCOVID19](https://kawalcovid19.id/)<br/>
Diperbarui pada: {{ date }}<br/>
Metode dapat dilihat di [sini]({{ '{{' }} site.baseurl {{ '}}' }}/2020/04/17/COVID19-has-the-curve-flatten/)
{% for place in places %}
#### {{ place['name'] }}

Hasil: {{ place['flatcurve_result'] }}

<img title="{{ place['name'] }}" src="{{ '{{' }} site.baseurl {{ '}}' }}/assets/idcovid19-daily/{{ place['dataid'] }}.png" width="100%"/>
{% endfor %}
