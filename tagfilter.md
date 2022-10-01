---

title: Tag Filter

---

<h2>Post Tags</h2>
<ul id="postTags" style="display:none">
{% assign rawtags = "" %}
{% for post in site.posts %}
  {% assign ttags = post.tags | join:'|' | append:'|' %}
  {% assign rawtags = rawtags | append:ttags %}
<li class="post">
<a href ="{{ post.url }}"> {{ post.title }} </a>
  {% for tagName in post.tags %}
<a href='/tags?tagName={{ tagName }}'><i class='glyphicon glyphicon-tag'></i>{{ tagName }}</a>
  {% endfor %}
</li>
{% endfor %}
{% for post in site.pages %}
  {% assign ttags = post.tags | join:'|' | append:'|' %}
  {% assign rawtags = rawtags | append:ttags %}
<li class="page"> 
<a href ="{{ post.url }}"> {{ post.title }} </a>
  {% for tagName in post.tags %}
<a href='/tags?tagName={{ tagName }}'><i class='glyphicon glyphicon-tag'></i>{{ tagName }}</a>
  {% endfor %}
</li>
{% endfor %}
{% assign rawtags = rawtags | split:'|' | sort %}
</ul>

<input type="text" id="tagInput" list="tagDataList">
<datalist id="tagDataList">

    <option value="tag1" />
    <option value="tag2" />
    <option value="tag3" />

</datalist>
</input>
<button id="butAdd" >Add</button>
<button id="submit" >submit</button>

<ul id="filterTags" >

    <li>Tags:</li>

</ul>

<hr>

<ul id="results">

    <li> <a href="#">post 1</a> </li>
    <li> <a href="#">post 2</a> </li>
    <li> <a href="#">post 3</a> </li>
    <li> <a href="#">post 4</a> </li>
    <li> <a href="#">post 5</a> </li>

</ul>

<style>

    #filterTags > li {
        display: inline;
        padding: 8px;
    }

</style>

<script >

    let dict = {};

    let tagInput = document.getElementById("tagInput");
        let submit = document.getElementById("submit");
        let butAdd = document.getElementById("butAdd");
        let filterTags = document.getElementById("filterTags");
        let result = document.getElementById("results");
        let tags = document.getElementsByClassName("tag");

        butAdd.addEventListener("click", (event) => {
            console.log("click");
            if(tagInput.value==="") {
                return;
            }
            let li = document.createElement("li");
            let liText = document.createTextNode(tagInput.value);
            li.appendChild(liText);
            li.addEventListener("click", tagEventHandler);
            filterTags.appendChild(li);
            tagInput.value = "";

        });

        submit.addEventListener("click", function () {
            result.innerHTML = "";
            let lis = filterTag.findElementsByTagName("li");
            var res=[];
            for(var i = 1; i< lis.lenght; i++) {
                let tag = lis[i].textContent;
                let arr = dict[tag];
                res = intersect(arr, res);
            }
            for(let item of res) {
                result.appendChild(createPostWithLink(item.title, item.url));
            }

        });

        function createPostWithLink(title, link) {
            let li = document.createElement("li");
            li.classList.add("post");
            li.innerHTML='<a href="' + link + '" >' + title + '</a>';
            return li;
        }
        // for(let element of tags) {
        //     element.addEventListener("click", tagEventHandler);
        // }
        function tagEventHandler() {
            console.log(this+"tag")
            this.remove();
        }

        function parseData() {
            let pages = document.getElementsByClassName("page");
            for(let page of pages) {

                let as = page.getElementsByTagName("a");
                if(as.length == 0) {
                    continue;
                }
                let postUrl = as[0].href;
                let postTitle = as[0].textContent;

                for(var i = 1; i < as.length; i++) {
                    let a = as[i];
                    let href = a.href;
                    let tagText = a.textContent;
                    if (dict[tagText] == undefined ) {
                        dict[tagText] = [];
                    }
                    dict[tagText].push({
                        "title": postTitle,
                        "url": postUrl
                    });
                }
            }
        }
    function intersect(a, b) {
        var setA = new Set(a);
        var setB = new Set(b);
        var intersection = new Set([...setA].filter(x => setB.has(x)));
        return Array.from(intersection);
    }

    parseData();

</script>
