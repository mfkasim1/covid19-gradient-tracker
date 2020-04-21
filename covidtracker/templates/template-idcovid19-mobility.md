---
layout: page
short_title:
title: Seberapa banyakkah pengurangan mobilitas antar provinsi?
permalink: /idcovid19-mobility/
---

Sumber data: [Facebook GeoInsight](https://www.facebook.com/geoinsights-portal/) dan [Indonesia GeoJSON](https://github.com/superpikar/indonesia-geojson) <br/>
Diperbarui pada: {{ date }}<br/>
{% for place in places %}
#### {{ place['name'] }}

Rata-rata 10 hari terakhir: \\(\left({{ place['mean_10_days'] }}\pm {{ place['std_10_days'] }}\right)\%\\)
{% if psbb in place %}

{% endif %}
<img title="{{ place['name'] }}" src="{{ '{{' }} site.baseurl {{ '}}' }}/assets/idcovid19-mobility/{{ place['fname'] }}" width="50%"/>
{% endfor %}
