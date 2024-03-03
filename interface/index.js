renderTable(data)

function renderTable(data, inverse=false) {
    let tbody = document.querySelector(".styled-table tbody");
    tbody.innerHTML = "";

    if (inverse) {
        for (let i = data.length-1; 0 <= i; i--) {
            appendRow(tbody, data[i]);
        }
    }
    else {
        for (let i = 0; i < data.length; i++) {
            appendRow(tbody, data[i]);
        }
    }
}

function appendRow(tbody, item) {
    let row = document.createElement('tr');
    let content = "<td>" + item.name + "</td>"
    content += "<td>" + item.id + "</td>";
    content += "<td>" + item.first + "</td>";
    content += "<td>" + item.last + "</td>";
    row.innerHTML = content;
    tbody.appendChild(row);
}