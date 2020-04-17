---
layout: page
short_title: COVID-19 (ID)
title:
permalink: /idcovid19/
---

## Apakah kurva telah melandai?

Sumber data: KawalCOVID19
{% for place in places %}
#### {{ place['name'] }}

Hasil: {{ place['flatcurve_result'] }}

<img title="{{ place['name'] }}" src="{{ '{{' }} site.baseurl {{ '}}' }}/assets/idcovid19-daily/{{ place['dataid'] }}.png" width="100%"/>
{% endfor %}
