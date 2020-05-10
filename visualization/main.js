var USE_MULTILINGUAL = true;

const LANGS = ["en", "fr"]// ["ar", "cz", "de", "es", "en", "fa", "fi", "fr", "id", "lv", "zh"]
const VISUALIZATION_SIZE = 0.7;
const VIEW_SIZE = 1.2;

var margin = {top: 20, right: 20, bottom: 20, left: 40},
    width = window.innerHeight * .8
    height = window.innerHeight * .8

/*
 * value accessor - returns the value to encode for a given data object.
 * scale - maps value to a visual display encoding, such as a pixel position.
 * map function - maps from data value to display value
 * axis - sets up axis
 */

// setup x
var xValue = function(d) { return d.x0;}, // data -> value
    xScale = d3.scaleLinear().domain([-60, 55]).range([0, width]), // value -> display
    xMap = function(d) { return xScale(xValue(d));}, // data -> display
    xAxis = d3.axisBottom(xScale);

// setup y
var yValue = function(d) { return d.x1;}, // data -> value
    yScale = d3.scaleLinear().domain([-60, 55]).range([height, 0]), // value -> display
    yMap = function(d) { return yScale(yValue(d));}, // data -> display
    yAxis = d3.axisLeft(yScale);

// Version 4.0
const pSBC=(p,c0,c1,l)=>{
    let r,g,b,P,f,t,h,i=parseInt,m=Math.round,a=typeof(c1)=="string";
    if(typeof(p)!="number"||p<-1||p>1||typeof(c0)!="string"||(c0[0]!='r'&&c0[0]!='#')||(c1&&!a))return null;
    if(!this.pSBCr)this.pSBCr=(d)=>{
        let n=d.length,x={};
        if(n>9){
            [r,g,b,a]=d=d.split(","),n=d.length;
            if(n<3||n>4)return null;
            x.r=i(r[3]=="a"?r.slice(5):r.slice(4)),x.g=i(g),x.b=i(b),x.a=a?parseFloat(a):-1
        }else{
            if(n==8||n==6||n<4)return null;
            if(n<6)d="#"+d[1]+d[1]+d[2]+d[2]+d[3]+d[3]+(n>4?d[4]+d[4]:"");
            d=i(d.slice(1),16);
            if(n==9||n==5)x.r=d>>24&255,x.g=d>>16&255,x.b=d>>8&255,x.a=m((d&255)/0.255)/1000;
            else x.r=d>>16,x.g=d>>8&255,x.b=d&255,x.a=-1
        }return x};
    h=c0.length>9,h=a?c1.length>9?true:c1=="c"?!h:false:h,f=this.pSBCr(c0),P=p<0,t=c1&&c1!="c"?this.pSBCr(c1):P?{r:0,g:0,b:0,a:-1}:{r:255,g:255,b:255,a:-1},p=P?p*-1:p,P=1-p;
    if(!f||!t)return null;
    if(l)r=m(P*f.r+p*t.r),g=m(P*f.g+p*t.g),b=m(P*f.b+p*t.b);
    else r=m((P*f.r**2+p*t.r**2)**0.5),g=m((P*f.g**2+p*t.g**2)**0.5),b=m((P*f.b**2+p*t.b**2)**0.5);
    a=f.a,t=t.a,f=a>=0||t>=0,a=f?a<0?t:t<0?a:a*P+t*p:0;
    if(h)return"rgb"+(f?"a(":"(")+r+","+g+","+b+(f?","+m(a*1000)/1000:"")+")";
    else return"#"+(4294967296+r*16777216+g*65536+b*256+(f?m(a*255):0)).toString(16).slice(1,f?undefined:-2)
}



// setup fill color
//var cValue = function(d) {return d.label;},
//    color = d3.scaleOrdinal(d3.schemeCategory10);

const USE_ALL = false;

colorMap = {
   "case": "#2e44c3", // "#152894"
   // "nmod": "#c2b5b5",
   "det": "#338833",
   "nsubj": "#dbb02e",
   // "obl": "#853ab4", //"#ba95d1",
   "amod": "#8f0909",
   "obj": "#3997c6", // "#853ab4",
   "advmod": "#d95cc6",
   "cc": "#696969",
   "conj": "#ff8400",
	 // "mark": "#9ccd79",
   // "iobj": "#222222"
   // "obl": "#ba95d1",
   // "xcomp": "#39ce78",
   // "obj": "#853ab4",
   // "acl": "#db5f57",
   // "advcl": "#db7557",
   // "advmod": "#db8b57",
   // "amod": "#dba157",
   // "appos": "#dbb757",
   // "aux": "#dbcd57",
   // "case": "#d3db57",
   // "cc": "#bddb57",
   // "ccomp": "#a7db57",
   // "clf": "#91db57",
   // "compound": "#7bdb57",
   // "conj": "#65db57",
   // "cop": "#57db5f",
   // "csubj": "#57db75",
   // "dep": "#57db8b",
   // "det": "#57dba1",
   // "discourse": "#57dbb7",
   // "dislocated": "#57dbcd",
   // "expl": "#57d3db",
   // "fixed": "#57bddb",
   // "flat": "#57a7db",
   // "goeswith": "#5791db",
   // "iobj": "#577bdb",
   // "list": "#5765db",
   // "mark": "#5f57db",
   // "nmod": "#7557db",
   // "nsubj": "#8b57db",
   // "nummod": "#a157db",
   // "obj": "#b757db",
   // "obl": "#cd57db",
   // "orphan": "#db57d3",
   // "parataxis": "#db57bd",
   // "punct": "#db57a7",
   // "reparandum": "#db5791",
   // "vocative": "#db577b",
   // "xcomp": "#db5765",
   //
   //  "reparandum": "#e6194b",
 // "fr-cop": "#d95cc6",
    //    "en-cop": "#91218d",
    //    "fr-xcomp": "#12776c",
    //    "en-xcomp": "#25e5dd",
    //    "fr-obl": "#5e910b",
    //    "en-obl": "#cbde36",
    //    "fr-nmod": "#ba1273",
    //    "en-nmod": "#831b57",
}

function colorFunc(d) {
	const USE_TWO = true;
	if (USE_TWO) {
		return (d.lang == "fr" ? darkColors : lightColors)[d.label];
	}	else {
		return colorMap[d.label];
	}
}
// function(d) { return colorMap[d.label];}

darkColors = {
    "det": "#338833",
    "amod": "#8f0909",
    "nsubj": "#dbb02e",
    "case": "#152894",
    "conj": "#ff8400",
    "cc": "#696969",
    "advmod": "#91218d",
    "obj": "#3997c6"
}

lightColors = {
    "det": "#88ee88",
    "amod": "#db5e35",
    "nsubj": "#a65b11",
    "case": "#5f72de",
    "conj": "#964f02",
    "cc": "#bdbdbd",
    "advmod": "#d95cc6",
    "obj": "#18658b"
}
// add the graph canvas to the body of the webpage
var svg = d3.select("#svg")
.attr("width", width + margin.left + margin.right)
.attr("height", height + margin.top + margin.bottom)
.append("g")
.attr("transform", "translate(" + margin.left + "," + margin.top + ")");

// add the tooltip area to the webpage
var tooltip = d3.select("body").append("div")
.attr("class", "sentenceTooltip")
.style("opacity", 0);

var dataField = d3.select(".field");

svg.append("g")
    .attr("class", "x axis")
    .attr("transform", "translate(0," + height + ")")
    .call(xAxis)
    .append("text")
    .attr("class", "label")
    .attr("x", width)
    .attr("y", -6)
    .style("text-anchor", "end")
    .text("x0");

// y-axis
svg.append("g")
    .attr("class", "y axis")
    .call(yAxis)
    .append("text")
    .attr("class", "label")
    .attr("transform", "rotate(-90)")
    .attr("y", 6)
    .attr("dy", ".71em")
    .style("text-anchor", "end")
    .text("x1");

console.log('svg loaded');
// load data
function loadData(langs) {
    svg.selectAll(".dot").remove();
    // loader settings
    var opts = {
      lines: 9, // The number of lines to draw
      length: 9, // The length of each line
      width: 5, // The line thickness
      radius: 14, // The radius of the inner circle
      color: '#EE3124', // #rgb or #rrggbb or array of colors
      speed: 1.9, // Rounds per second
      trail: 40, // Afterglow percentage
      className: 'spinner', // The CSS class to assign to the spinner
    };
    var target = document.getElementById("visualization");
    console.log(target);
    var spinner = new Spinner(opts).spin(target);
    var spinnerElem = document.getElementsByClassName('spinner')[0];
    spinnerElem.style.left = "25%";
    var loading = document.getElementById("loading");
    loading.style.opacity = 1.0;
    d3.tsv("dev.tsv", (d) => {
        let idx = parseInt(d.idx);
        let newHTML = d.sentence.split(' ');
        newHTML.splice(idx, 0, "<b>");
        newHTML.splice(idx+2, 0, "</b>");
        newHTML = newHTML.join(" ");
        newHTML += "<br/>" + d.relation;
        newHTML += `<br/> (${+d.x0} ${+d.x1})`;
        let label = d.relation;
        if (label.includes(":")) label = label.substr(0, label.indexOf(':'));
        return {
            x0: +d.x0,
            x1: +d.x1,
            label: label,
            sentence: d.sentence,
            idx: idx,
            html: newHTML,
            dot_id: "dot-" + d.x0.replace('.', '_'),
            lang: d.lang,
        }
    }).then((data) => {

        data = data.filter((d) => {
            return (langs.includes(d.lang) && colorMap.hasOwnProperty(d.label));
        });
        d3.shuffle(data);


        svg.selectAll(".dot")
            .data(data)
            .enter().append("circle")
            .attr("class", "dot")
            .style("fill", "purple")
            .attr("r", VIEW_SIZE)
            .style("fill", colorFunc)
            .attr("cx", xMap)
            .attr("cy", yMap)
            .attr("id", function(d) { return d.dot_id; })
            .on("mouseover", function(d) {
                tooltip.transition()
                    .duration(200)
                    .style("opacity", 1);
                tooltip.html(d.html)
                    .style("left", (d3.event.pageX + 5) + "px")
                    .style("top", (d3.event.pageY - 28) + "px");
            })
            .on("mouseout", function(d) {
                tooltip.transition()
                    .duration(500)
                    .style("opacity", 0);
            })
            .on("click", function(d) {
                svg.select(".selectedDot")
                    .attr("r", 1)
                    .classed("selectedDot", false);
                svg.select("#" + d.dot_id)
                    .attr("r", 10)
                    .classed("selectedDot", true);
                dataField.html(d.html);
            });
        spinner.stop();
        loading.style.opacity = 0;
    });
}

const checkboxes = document.getElementById('language-checkboxes');
for (const lang of LANGS) {
    const newCheckbox = `<input type="checkbox" id="check-${lang}" class="checkbox" checked> `;
    checkboxes.insertAdjacentHTML('beforeend', newCheckbox);
    const newLabel = `<label class="form-check-label" for="exampleCheck1">${lang}</label><br/>`;
    checkboxes.insertAdjacentHTML('beforeend', newLabel);
    console.log(newCheckbox);
}

const legend = document.getElementById('legend');
for (const label in colorMap) {
    if (colorMap.hasOwnProperty(label)) {
        const line = `<p class="legendRow"><span class="dot" style="background-color:${colorMap[label]}"></span> ${label}`;
        console.log(line);
        legend.insertAdjacentHTML('beforeend', line);
    }
}
// <span class="dot"></span> xcomp

loadData(LANGS);



function reloadData() {
    loadData(LANGS.filter((lang) => {
        return document.getElementById(`check-${lang}`).checked;
    }));
}

function selectAll() {
    LANGS.forEach((lang) => {
        document.getElementById(`check-${lang}`).checked = true;
    });
}

function saveImage() {
    //get svg element.
    var svg = document.getElementById("svg");

    //get svg source.
    var serializer = new XMLSerializer();
    var source = serializer.serializeToString(svg);

    //add name spaces.
    if(!source.match(/^<svg[^>]+xmlns="http\:\/\/www\.w3\.org\/2000\/svg"/)){
        source = source.replace(/^<svg/, '<svg xmlns="http://www.w3.org/2000/svg"');
    }
    if(!source.match(/^<svg[^>]+"http\:\/\/www\.w3\.org\/1999\/xlink"/)){
        source = source.replace(/^<svg/, '<svg xmlns:xlink="http://www.w3.org/1999/xlink"');
    }

    //add xml declaration
    source = '<?xml version="1.0" standalone="no"?>\r\n' + source;

    //convert svg source to URI data scheme.
    var url = "data:image/svg+xml;charset=utf-8,"+encodeURIComponent(source);

    //set url value to a element's href attribute.
    document.getElementById("link").href = url;
    //you can download svg file by right click menu.
}
