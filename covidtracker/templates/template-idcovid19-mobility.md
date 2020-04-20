---
layout: page
short_title: IDCOVID-19 (Movement)
title:
permalink: /idcovid19-movement/
---

## Seberapa banyakkah pengurangan mobilitas antar provinsi?

Sumber data: [Facebook GeoInsight](https://www.facebook.com/geoinsights-portal/) dan [Indonesia GeoJSON](https://github.com/superpikar/indonesia-geojson) <br/>
Diperbarui pada: {{ date }}<br/>
{% for place in places %}
#### {{ place['name'] }}

<img title="{{ place['name'] }}" src="{{ '{{' }} site.baseurl {{ '}}' }}/assets/idcovid19-mobility/{{ place['fname'] }}" width="50%"/>
{% endfor %}
