---
title: "datascience_basics"
layout: archive
permalink: categories/basics
author_profile: true
sidebar_main: true
---


{% assign posts = site.categories.basics %}
{% for post in posts %} {% include archive-single.html type=page.entries_layout %} {% endfor %}
