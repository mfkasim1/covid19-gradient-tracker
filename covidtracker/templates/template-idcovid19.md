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

Probabilitas menurun: \\({{ place['decline_prob'] }}\%\\)<br/>
Hasil: {{ place['flatcurve_result'] }}<br/>
{% if place['total_cases_median'] != "" %}Estimasi jumlah kasus: {{ place['total_cases_median'] }} (95% CI: {{ place['total_cases_025'] }} - {{ place['total_cases_975'] }})
{% endif %}
<img title="{{ place['name'] }}" src="{{ '{{' }} site.baseurl {{ '}}' }}/assets/idcovid19-daily/{{ place['dataid'] }}.png" width="100%"/>
{% endfor %}
